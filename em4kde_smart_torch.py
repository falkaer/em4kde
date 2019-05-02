import numpy as np
import torch

def exclude_mask(N, i):
    mask = torch.ones(N, dtype=torch.uint8)
    mask[i] = 0
    
    return mask

eps = 1e-10

class KDE:
    def __init__(self, X, strategy):
        self.strategy = strategy
        self.X = X
        
        # self.sigma = np.eye(dims)
        self.sigma = self.init_sigma_random_cov()
        # TODO: constrain sigma to be diagonal for regularization
    
    def init_sigma_random_cov(self):
        
        N, D = self.X.shape
        A = torch.rand(D, D, device=self.X.device)
        sign, logdet = torch.slogdet(A)
        
        # must be definite
        if logdet.abs() < 0.00001:
            return self.init_sigma_random_cov()
        
        return A @ A.t()
    
    # @profile
    def e_step(self, Q_train, Q_test, A):
        
        N_train, D = Q_train.shape
        N_test, D = Q_test.shape
        gamma = torch.empty(N_train, N_test, device=self.X.device)
        
        for i in range(N_train):
            gamma[i] = cholesky_multivariate_normal(Q_test, Q_train[i], A)
        
        gammasum = gamma.sum(dim=0)
        gammasum[gammasum == 0] = 1
        
        gamma /= gammasum
        return gamma
    
    def m_step(self, gamma):
        
        N, D = self.X.shape
        sigma = torch.zeros(D, D, device=self.X.device)
        mask = ~torch.isnan(gamma)
        
        for i in range(N):
            
            diff = self.X[mask[i]] - self.X[i]
            gammai = gamma[i, mask[i]]
            gamma_sum = gammai.sum()
            
            if gamma_sum != 0:
                sigma += (gammai[None].t() * diff).t() @ diff / gamma_sum
        
        return sigma / N
    
    def step(self):
        
        N, D = self.X.shape
        A = torch.cholesky(self.sigma)
        Ainv = torch.inverse(A)
        
        splits = self.strategy.get_splits()
        gamma = torch.full((N, N), np.nan, device=self.X.device)
        
        for train_idx, test_idx in splits:
            Q_train = self.X[train_idx] @ Ainv.t()
            Q_test = self.X[test_idx] @ Ainv.t()
            
            gamma[train_idx[:, None], test_idx] = self.e_step(Q_train, Q_test, A)
        
        self.sigma = self.m_step(gamma)
    
    def density(self, X, Y):
        
        N, D = X.shape
        
        A = torch.cholesky(self.sigma)
        Ainv = torch.inverse(A)
        
        Q_X = X @ Ainv.t()
        Q_Y = Y @ Ainv.t()
        
        return torch.stack([cholesky_multivariate_normal(Q_Y, Q_X[i], A) for i in range(N)]).sum(dim=0) / N
    
    def log_likelihood(self, X):
        
        N, D = X.shape
        A = torch.cholesky(self.sigma)
        Ainv = torch.inverse(A)
        
        return -torch.stack(
                    [torch.log(eps + cholesky_multivariate_normal(X[exclude_mask(N, i)] @ Ainv.t(), X[i] @ Ainv.t(), A)) for i
                     in range(N)]).sum() / N

def cholesky_multivariate_normal(Q_test, q_train, A):
    detA = A.diagonal().prod()
    M = A.shape[0]
    
    coeff = 1 / (2 * np.pi ** (M / 2) * detA)
    # exp = -1 / 2 * (torch.sum(Q_test ** 2, dim=1) + q_train @ q_train - 2 * Q_test @ q_train)
    exp = -1 / 2 * (torch.sum(Q_test ** 2, dim=1) + q_train @ q_train - 2 * Q_test @ q_train)
    
    return coeff * torch.exp(exp)

def plot_kde(kde, X):
    import matplotlib.pyplot as plt
    
    Xn = X.cpu().numpy()
    x, y = Xn[:, 0], Xn[:, 1]
    
    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, 
                      y.min():y.max():y.size ** 0.5 * 1j]
    
    Y = torch.from_numpy(np.vstack([xi.flatten(), yi.flatten()]).T).float().to(X.device)
    zi = kde.density(X, Y)
    
    plt.figure(figsize=(7, 8))
    plt.contourf(xi, yi, zi.reshape(xi.shape))
    
    for train_idx, test_idx in kde.strategy.get_splits():
        X_train = Xn[train_idx]
        X_test = Xn[test_idx]
        
        plt.plot(X_train[:, 0], X_train[:, 1], '.', color='blue')
        plt.plot(X_test[:, 0], X_test[:, 1], '.', color='red')
    
    plt.gca().set_xlim(x.min(), x.max())
    plt.gca().set_ylim(y.min(), y.max())
    
    plt.show()

def plot_training_progress(prog):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(16, 9))
    
    plt.plot(prog)
    
    plt.title("Log likelihood")
    plt.xlabel("Iterations")
    plt.ylabel("Log likelihood")
    
    plt.show()

def train(kde, X, iterations):
    likelihoods = []
    
    for iteration in range(iterations):
        
        kde.step()
        likelihoods.append(kde.log_likelihood(X))
        
        if iteration % 3 == 0:
            # plot_kde(kde, X)
            print(iteration, likelihoods[-1].item())
    
    # plot_training_progress(likelihoods)
