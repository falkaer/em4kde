import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal

class KDE:
    def __init__(self, dims):
        self.sigma = np.eye(dims)
    
    def e_step(self, X_train, X_test, sigma):
    
        N_train, D = X_train.shape
        N_test, _ = X_test.shape
        pi = 1 / N_train

        gamma = np.empty((N_train, N_test))
        div = [pi * multivariate_normal.pdf(X_test, mean=X_train[j], cov=sigma) for j in range(N_train)]
        
        for n in range(N_train):
            gamma[n] = pi * multivariate_normal.pdf(X_test, mean=X_train[n], cov=sigma) / sum(div)
        
        return gamma
        
    def m_step(self, X_train, X_test, gamma):
        
        N_train, D = X_train.shape
        sigma = np.zeros((D, D))
        
        for n in range(N_train):
            prod = X_test - X_train[n]

            sigma += 1 / gamma[n].sum() * (gamma[n].reshape(1, -1).T * prod).T @ prod
        
        return np.divide(sigma, N_train)
    
    def step(self, X_train, X_test):
        
        gamma = self.e_step(X_train, X_test, self.sigma)
        self.sigma = self.m_step(X_train, X_test, gamma)
    
    def density(self, X_train, Y):
    
        N_train, D = X_train.shape
        return np.array([multivariate_normal.pdf(Y, mean=X_train[n], cov=self.sigma) for n in range(N_train)]).sum(axis=0) / N_train
        
    def log_likelihood(self, X_train, X_test):
    
        N_train, D = X_train.shape
        N_test, _ = X_test.shape
        pi = 1 / N_train
        
        return -np.log([pi * multivariate_normal.pdf(X_test, mean=X_train[n], cov=self.sigma) for n in range(N_train)]).sum()

def plot_kde(kde, X_train, X_test):
    
    import matplotlib.pyplot as plt

    X = np.vstack((X_train, X_test))
    x, y = X[:, 0], X[:, 1]
    
    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
    zi = kde.density(X_train, np.vstack([xi.flatten(), yi.flatten()]).T)
    
    fig = plt.figure(figsize=(7, 8))
    plt.contourf(xi, yi, zi.reshape(xi.shape))
    
    plt.plot(X[:, 0], X[:, 1], '.')
    
    plt.gca().set_xlim(x.min(), x.max())
    plt.gca().set_ylim(y.min(), y.max())
    
    plt.show()

def train(kde, X):
    
    max_iter = 100
    
    N, D = X.shape
    X_train = X[0:N // 2, :]
    X_test = X[N // 2:, :]
    
    for iteration in range(max_iter):
        
        kde.step(X_train, X_test)
        print(iteration, kde.log_likelihood(X_train, X_test))
        
        if iteration % 10 == 0:
            plot_kde(kde, X_train, X_test)
        
    
X = loadmat('clusterdata2d.mat')['data']
np.random.shuffle(X)

N, D = X.shape
kde = KDE(D)

train(kde, X)
