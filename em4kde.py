import numpy as np
from scipy.stats import multivariate_normal

def exclude_mask(N, i):
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    
    return mask

class KDE:
    def __init__(self, strategy, dims):
        self.strategy = strategy
        self.sigma = np.eye(dims)

    #@profile
    def e_step(self, X_train, X_test, sigma):
        
        N_train, D = X_train.shape
        N_test, D = X_test.shape
        gamma = np.empty((N_train, N_test))
        
        for i in range(N_train):
            gamma[i] = multivariate_normal.pdf(X_test, mean=X_train[i], cov=sigma)
        
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
        
        N, D = X.shape
        splits = self.strategy.get_splits()
        sigma = np.zeros((D, D))
        
        for X_train, X_test in splits:
            
            gamma = self.e_step(X_train, X_test, self.sigma)
            sigma += self.m_step(X_train, X_test, gamma)
        
        self.sigma = sigma / len(splits)
        
    def density(self, X, Y):
        
        N, D = X.shape
        return np.array([multivariate_normal.pdf(Y, mean=X[n], cov=self.sigma) for n in range(N)]).sum(axis=0) / N
    
    def log_likelihood(self, X):
        
        N, D = X.shape
        return -np.array([multivariate_normal.logpdf(X[exclude_mask(N, i)], mean=X[i], cov=self.sigma) for i in
                          range(N)]).sum() / N

def plot_kde(kde, X):
    import matplotlib.pyplot as plt
    
    x, y = X[:, 0], X[:, 1]
    
    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
    zi = kde.density(X, np.vstack([xi.flatten(), yi.flatten()]).T)
    
    plt.figure(figsize=(7, 8))
    plt.contourf(xi, yi, zi.reshape(xi.shape))
    
    plt.gca().set_xlim(x.min(), x.max())
    plt.gca().set_ylim(y.min(), y.max())
    
    plt.show()

def train(kde, X, iterations):
    
    for iteration in range(iterations):
        
        kde.step(X)
        print(iteration, kde.log_likelihood(X))
        
        if iteration % 3 == 0:
            plot_kde(kde, X)
