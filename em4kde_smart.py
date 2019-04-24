import numpy as np
import torch

def exclude_mask(N, i):
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    
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
        A = np.random.rand(D, D)
        sign, logdet = np.linalg.slogdet(A)
        
        # must be definite
        if abs(logdet) < 0.00001:
            return self.init_sigma_random_cov()
        
        return A @ A.T
    
    # @profile
    def e_step(self, Q_train, Q_test, A):
        
        N_train, D = Q_train.shape
        N_test, D = Q_test.shape
        gamma = np.empty((N_train, N_test))
        
        for i in range(N_train):
            gamma[i] = cholesky_multivariate_normal(Q_test, Q_train[i], A)
        
        gamma /= gamma.sum(axis=0)
        
        return gamma
    
    def m_step(self, gamma, train_len):
        
        N, D = self.X.shape
        sigma = np.zeros((D, D))
        mask = ~np.isnan(gamma)
        
        for i in range(N):
            
            prod = self.X[mask[i]] - self.X[i]
            gammai = gamma[i, mask[i]]
            sigma += 1 / gammai.sum() * (gammai[np.newaxis].T * prod).T @ prod #/ len(gammai)
        
        #TODO: pls
        # sigma = 1 / gamma.sum() * (gamma[np.newaxis].T * prod).T @ prod
        
        return sigma / N
    
    def step(self):
    
        N, D = self.X.shape
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)
        
        splits = self.strategy.get_splits()
        lens = self.strategy.train_len()
        
        gamma = np.full((N, N), np.nan)
        
        for train_idx, test_idx in splits:
            
            Q_train = self.X[train_idx] @ Ainv.T
            Q_test = self.X[test_idx] @ Ainv.T
            
            gamma[train_idx[:, np.newaxis], test_idx] = self.e_step(Q_train, Q_test, A)
            
        self.sigma = self.m_step(gamma, lens)
    
    def density(self, X, Y):
        
        N, D = X.shape
        
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)
        
        Q_X = X @ Ainv.T
        Q_Y = Y @ Ainv.T
        
        return np.array([cholesky_multivariate_normal(Q_Y, Q_X[i], A) for i in range(N)]).sum(axis=0)
    
    def log_likelihood(self, X):
        
        N, D = X.shape
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)
        
        return -np.array(
                    [np.log(eps + cholesky_multivariate_normal(X[exclude_mask(N, i)] @ Ainv.T, X[i] @ Ainv.T, A)) for i
                     in range(N)]).sum() / N

def cholesky_multivariate_normal(Q_test, q_train, A):
    detA = A.diagonal().prod()
    M = A.shape[0]
    
    coeff = 1 / (2 * np.pi ** (M / 2) * detA)
    exp = -1 / 2 * (np.sum(Q_test ** 2, axis=1) + q_train.T @ q_train - 2 * Q_test @ q_train)
    
    return coeff * np.exp(exp)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / np.sum(e_x, axis=0)

def plot_kde(kde, X):
    import matplotlib.pyplot as plt
    
    x, y = X[:, 0], X[:, 1]
    
    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
    zi = kde.density(X, np.vstack([xi.flatten(), yi.flatten()]).T)
    
    plt.figure(figsize=(7, 8))
    plt.contourf(xi, yi, zi.reshape(xi.shape))
    
    for train_idx, test_idx in kde.strategy.get_splits():
        X_train = X[train_idx]
        X_test = X[test_idx]
        
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
        print(iteration, likelihoods[-1])
        
        if iteration % 3 == 0:
            plot_kde(kde, X)
    
    plot_training_progress(likelihoods)

def m_step(X, gamma):
    
    N, D = X.shape
    sigma = np.zeros((D, D))
    
    for i in range(N):
        prod = X - X[i]
        awf = (gamma[i, np.newaxis].T * prod).T @ prod
        print(awf)
        sigma += 1 / gamma[i].sum() * awf
    
    return sigma / N

def sig(X, x, gammai):
    prod = X - x
    awf = (gammai[np.newaxis].T * prod).T @ prod
    return 1 / gammai.sum() * awf