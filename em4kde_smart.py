import numpy as np
from scipy.stats import multivariate_normal

def exclude_mask(N, i):
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    
    return mask

eps = 1e-10

class KDE:
    
    def __init__(self, strategy, dims):
        self.strategy = strategy
        self.dims = dims
        
        self.sigma = np.eye(dims)
    
    # @profile
    def e_step(self, Q_train, Q_test, A):
        
        N_train, D = Q_train.shape
        N_test, D = Q_test.shape
        gamma = np.empty((N_train, N_test))
        
        for i in range(N_train):
            gamma[i] = cholesky_multivariate_normal(Q_test, Q_train[i], A)
        
        gamma /= gamma.sum(axis=0)
        
        return gamma
    
    def m_step(self, X_train, X_test, gamma):
        
        N_train, D = X_train.shape
        N_test, D = X_test.shape
        sigma = np.zeros((D, D))
        
        for i in range(N_train):
            prod = X_test - X_train[i]
            sigma += 1 / gamma[i].sum() * (gamma[i, np.newaxis].T * prod).T @ prod
        
        return sigma / N_train
    
    def step(self, X):
        
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)
        
        N, D = X.shape
        splits = self.strategy.get_splits()
        sigma = np.zeros((D, D))
        
        for X_train, X_test in splits:
            Q_train = X_train @ Ainv.T
            Q_test = X_test @ Ainv.T
            
            gamma = self.e_step(Q_train, Q_test, self.sigma)
            sigma += self.m_step(X_train, X_test, gamma)
        
        self.sigma = sigma / len(splits)
    
    def density(self, X, Y):
        
        N, D = X.shape
        
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)
        
        Q_X = X @ Ainv.T
        Q_Y = Y @ Ainv.T

        return np.array([cholesky_multivariate_normal(Q_Y, Q_X[i], A) for i in range(N)]).sum(axis=0) / N
    
    def log_likelihood(self, X):
        
        N, D = X.shape
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)

        return -np.array([np.log(eps + cholesky_multivariate_normal(X[exclude_mask(N, i)] @ Ainv.T, X[i] @ Ainv.T, A)) for i in range(N)]).sum() / N
        
def cholesky_multivariate_normal(Q_test, q_train, A):
    detA = A.diagonal().prod()
    M = A.shape[0]
    
    coeff = 1 / (2 * np.pi ** (M / 2) * detA)
    exp = -1 / 2 * (np.sum(Q_test ** 2, axis=1) + q_train.T @ q_train - 2 * Q_test @ q_train)
    
    return coeff * np.exp(exp)

def plot_kde(kde, X):
    import matplotlib.pyplot as plt
    
    x, y = X[:, 0], X[:, 1]
    
    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
    zi = kde.density(X, np.vstack([xi.flatten(), yi.flatten()]).T)
    
    fig = plt.figure(figsize=(7, 8))
    plt.contourf(xi, yi, zi.reshape(xi.shape))
    
    plt.plot(X[:, 0], X[:, 1], '.')
    
    plt.gca().set_xlim(x.min(), x.max())
    plt.gca().set_ylim(y.min(), y.max())
    
    plt.show()

def train(kde, X, iterations):
    for iteration in range(iterations):
        
        kde.step(X)
        print(iteration, kde.log_likelihood(X))
        
        if iteration % 3 == 0:
            plot_kde(kde, X)
