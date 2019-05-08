import os
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

class KDE:
    
    def __init__(self, X, strategy):
        self.strategy = strategy
        self.X = X
        
        self.sigma = self.init_sigma_sample_cov()
        # self.sigma = self.init_sigma_random_cov()
    
    def init_sigma_sample_cov(self):
        return np.cov(self.X, rowvar=False)
    
    def init_sigma_random_cov(self):
        
        N, D = self.X.shape
        A = np.random.randn(D, D)
        AA = A @ A.T
        
        sign, logdet = np.linalg.slogdet(AA)
        
        # must be definite
        if abs(logdet) < 0.00001:
            return self.init_sigma_random_cov()
        
        return AA
    
    # @profile
    def e_step(self, Q_train, Q_test, A):
        
        N_train, D = Q_train.shape
        N_test, D = Q_test.shape
        log_gamma = np.empty((N_train, N_test))
        
        for i in range(N_train):
            log_gamma[i] = log_cholesky_multivariate_normal(Q_test, Q_train[i], A)
        
        log_gamma_sum = logsumexp(log_gamma, axis=0)
        log_gamma -= log_gamma_sum
        
        return log_gamma
    
    def m_step(self, log_gamma):
        
        N, D = self.X.shape
        sigma = np.zeros((D, D))
        mask = ~np.isnan(log_gamma)
        
        for i in range(N):
            diff = self.X[mask[i]] - self.X[i]
            log_gammai = log_gamma[i, mask[i]]
            log_gammai_sum = logsumexp(log_gammai)
            
            weighted_diff = np.exp(log_gammai)[None].T * diff
            div = np.exp(log_gammai_sum)
            
            if div != 0:
                sigma += weighted_diff.T @ diff / div
        
        return sigma / N
    
    def step(self):
        
        N, D = self.X.shape
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)
        
        splits = self.strategy.get_splits()
        log_gamma = np.full((N, N), np.nan)
        
        for train_idx, test_idx in splits:
            Q_train = self.X[train_idx] @ Ainv.T
            Q_test = self.X[test_idx] @ Ainv.T
            
            log_gamma[train_idx[:, None], test_idx] = self.e_step(Q_train, Q_test, A)
        
        self.sigma = self.m_step(log_gamma)
    
    def density(self, Y):
        
        N, D = self.X.shape
        
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)
        
        Q_X = self.X @ Ainv.T
        Q_Y = Y @ Ainv.T
        
        return np.exp(logsumexp([log_cholesky_multivariate_normal(Q_Y, Q_X[i], A) for i in range(N)], axis=0)) / N
    
    def log_likelihood(self, X):
        
        A = np.linalg.cholesky(self.sigma)
        Ainv = np.linalg.inv(A)
        
        N, _ = X.shape
        Q = X @ Ainv.T
        
        return np.sum(logsumexp(log_cholesky_multivariate_normal(Q[i], Q, A)) - np.log(N) for i in range(N))
    
    def save_kde(self, fname):
        np.savez(fname, sigma=self.sigma, X=self.X)

def load_kde(fname):
    npz = np.load(fname)
    kde = KDE(npz['X'], None)
    kde.sigma = npz['sigma']
    
    return kde

def log_cholesky_multivariate_normal(Q_test, Q_train, A):
    log_detA = np.log(A.diagonal()).sum()
    M = A.shape[0]
    
    log_coeff = - M / 2 * np.log(2 * np.pi) - log_detA
    exp = -1 / 2 * (pairwise_dot(Q_test, Q_test) + pairwise_dot(Q_train, Q_train) - 2 * pairwise_dot(Q_test, Q_train))
    
    return log_coeff + exp

def logsumexp(x, axis=None):
    
    x_max = np.max(x, axis=axis)
    
    with np.errstate(under='ignore'):
        return np.log(np.sum(np.exp(x - x_max), axis=axis)) + x_max

def pairwise_dot(X, Y):
    # pairwise dot product over row vectors of matrices, 
    # promotes vectors to matrices and broadcasts over singleton dimension
    return np.einsum('ij,ij->i', np.atleast_2d(X), np.atleast_2d(Y))

def plot_kde(kde):
    x, y = kde.X[:, 0], kde.X[:, 1]
    
    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
    zi = kde.density(np.vstack([xi.flatten(), yi.flatten()]).T)
    
    plt.figure(figsize=(7, 8))
    plt.contourf(xi, yi, zi.reshape(xi.shape))
    
    for train_idx, test_idx in kde.strategy.get_splits():
        X_train = kde.X[train_idx]
        X_test = kde.X[test_idx]
        
        plt.plot(X_train[:, 0], X_train[:, 1], '.', color='blue')
        plt.plot(X_test[:, 0], X_test[:, 1], '.', color='red')
    
    plt.gca().set_xlim(x.min(), x.max())
    plt.gca().set_ylim(y.min(), y.max())
    
    plt.show()

def plot_training_progress(prog):
    plt.figure(figsize=(16, 9))
    
    plt.plot(prog)
    
    plt.title("Log likelihood")
    plt.xlabel("Iterations")
    plt.ylabel("Log likelihood")
    
    plt.show()
    
    if os.environ.get('DISPLAY', '') == '':
        plt.savefig('log_likelihood.png')
        plt.clf()

def train(kde, iterations):
    likelihoods = []
    
    for iteration in range(iterations):
        
        kde.step()
        likelihoods.append(kde.log_likelihood(kde.X))
        print(iteration, likelihoods[-1])
        
        if kde.X.shape[1] == 2 and iteration % 3 == 0:
            plot_kde(kde)
    
    plot_training_progress(likelihoods)
