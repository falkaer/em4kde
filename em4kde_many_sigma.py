import numpy as np

def exclude_mask(N, i):
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    
    return mask

eps = 1e-10

class KDE:
    
    def __init__(self, X, strategy):
        
        self.X = X
        self.strategy = strategy
        
        # self.sigma = np.eye(dims)
        # self.sigmas = [self.init_sigma_random_cov() for _ in range(X.shape[0])]
        self.sigmas = [np.eye(X.shape[1]) for _ in range(X.shape[0])]
        # TODO: constrain sigma to be diagonal for regularization
    
    # is bad
    def init_sigma_random_gamma(self):
        
        splits = self.strategy.get_splits()
        
        N, D = self.X.shape
        sigma = np.zeros((D, D))
        
        for X_train, X_test in splits:
            A = np.random.rand(len(X_train), len(X_test))
            gamma = softmax(A)
            print(gamma.shape)
            
            sigma += self.m_step(X_train, X_test, gamma)
        
        return sigma / len(splits)
    
    # is good
    def init_sigma_random_cov(self):
        
        N, D = self.X.shape
        A = np.random.rand(D, D)
        sign, logdet = np.linalg.slogdet(A)
        
        # must be definite
        if abs(logdet) < 0.00001:
            return self.init_sigma_random_cov()
        
        return A @ A.T
    
    # @profile
    def e_step(self, X_train, X_test, As, Ainvs):
        
        N_train, D = X_train.shape
        N_test, D = X_test.shape
        gamma = np.empty((N_train, N_test))
        
        for i in range(N_train):
            Q_train = X_train[i] @ Ainvs[i].T
            Q_test = X_test @ Ainvs[i].T
            gamma[i] = cholesky_multivariate_normal(Q_test, Q_train, As[i])
        
        gamma /= gamma.sum(axis=0)
        
        return gamma
    
    def m_step(self, X_train, X_test, gamma):
        
        N_train, D = X_train.shape
        N_test, D = X_test.shape
        sigmas = np.zeros((N_train, D, D))
        
        for i in range(N_train):
            
            prod = X_test - X_train[i]
            sigmas[i] = (1 / gamma[i].sum() * (gamma[i, np.newaxis].T * prod).T @ prod)
        
        return sigmas
    
    def step(self):
        
        N, D = self.X.shape
        
        As = np.array([np.linalg.cholesky(self.sigmas[i]) for i in range(N)])
        Ainvs = np.array([np.linalg.inv(A) for A in As])
        
        splits = self.strategy.get_splits()
        sigmas = np.zeros((N, D, D))
        
        for train_idx, test_idx in splits:
            X_train = self.X[train_idx]
            X_test = self.X[test_idx]
            
            gamma = self.e_step(X_train, X_test, As[train_idx], Ainvs[train_idx])
            sigmas[train_idx] += self.m_step(X_train, X_test, gamma)
        
        self.sigmas = sigmas / len(splits)
        # self.sigmas = np.random.choice()
    
    def density(self, X, Y):
        
        N, D = X.shape
        N_out, _ = Y.shape
        
        As = np.array([np.linalg.cholesky(self.sigmas[i]) for i in range(N)])
        Ainvs = np.array([np.linalg.inv(A) for A in As])
        
        return np.array([cholesky_multivariate_normal(Y @ Ainvs[i].T, X[i] @ Ainvs[i].T, As[i]) for i in range(N)]).sum(axis=0) / N_out
        
    def log_likelihood(self, X):
        
        # N, D = X.shape
        # A = np.linalg.cholesky(self.sigma)
        # Ainv = np.linalg.inv(A)
        # 
        # return -np.array(
        #             [np.log(eps + cholesky_multivariate_normal(X[exclude_mask(N, i)] @ Ainv.T, X[i] @ Ainv.T, A)) for i
        #              in range(N)]).sum() / N
        return 0

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

def train(kde, iterations):
    likelihoods = []
    
    for iteration in range(iterations):
        
        kde.step()
        likelihoods.append(kde.log_likelihood(kde.X))
        print(iteration, likelihoods[-1])
        
        if iteration % 3 == 0:
            plot_kde(kde, kde.X)
    
    # plot_training_progress(likelihoods)
