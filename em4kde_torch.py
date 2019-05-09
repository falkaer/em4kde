import os
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
import torch

def exclude_mask(N, i):
    mask = torch.ones(N, dtype=torch.uint8)
    mask[i] = 0
    
    return mask

def atleast_2d(x):
    ndim = len(x.shape)
    
    if ndim == 0:
        return x[None, None]
    elif ndim == 1:
        return x[None]
    else:
        return x

def cov(X):
    N, _ = X.shape
    
    x_bar = X.mean(dim=0)
    diff = X - x_bar
    
    return diff.t() @ diff / (N - 1)

eps = 1e-10

class KDE:
    def __init__(self, X, strategy):
        self.strategy = strategy
        self.X = X
        
        # self.sigma = np.eye(dims)
        self.sigma = self.init_sigma_random_cov()
        self.sigma = self.init_sigma_sample_cov()
        # TODO: constrain sigma to be diagonal for regularization
    
    def init_sigma_random_cov(self):
        
        N, D = self.X.shape
        A = torch.rand(D, D, device=self.X.device)
        AA = A @ A.t()
        
        sign, logdet = torch.slogdet(AA)
        
        # must be definite
        if logdet.abs() < 0.00001:
            print('wtf')
            return self.init_sigma_random_cov()
        
        return AA
    
    def init_sigma_sample_cov(self):
        return cov(self.X)
    
    # @profile
    def e_step(self, Q_train, Q_test, A):
        
        N_train, D = Q_train.shape
        N_test, D = Q_test.shape
        log_gamma = torch.empty(N_train, N_test, device=self.X.device)
        
        for i in range(N_train):
            log_gamma[i] = log_cholesky_multivariate_normal(Q_test, Q_train[i], A)
        
        return log_gamma
    
    def m_step(self, log_gamma):
        
        N, D = self.X.shape
        sigma = torch.zeros(D, D, device=self.X.device)
        mask = ~torch.isnan(log_gamma)
        
        for i in range(N):
            
            diff = self.X[mask[i]] - self.X[i]
            log_gammai = log_gamma[i, mask[i]]
            log_gammai_sum = torch.logsumexp(log_gammai, dim=0)
            log_gammai -= log_gammai_sum
            
            weighted_diff = torch.exp(log_gammai)[None].t() * diff
            sigma += weighted_diff.t() @ diff
        
        return sigma / N
    
    def step(self):
        
        N, D = self.X.shape
        A = torch.cholesky(self.sigma)
        Ainv = torch.inverse(A)
        
        splits = self.strategy.get_splits()
        log_gamma = torch.full((N, N), np.nan, device=self.X.device)
        
        for train_idx, test_idx in splits:
            Q_train = self.X[train_idx] @ Ainv.t()
            Q_test = self.X[test_idx] @ Ainv.t()
            
            log_gamma[train_idx[:, None], test_idx] = self.e_step(Q_train, Q_test, A)
        
        self.sigma = self.m_step(log_gamma)
    
    def density(self, Y):
        
        N, D = self.X.shape
        
        A = torch.cholesky(self.sigma)
        Ainv = torch.inverse(A)
        
        Q_X = self.X @ Ainv.t()
        Q_Y = Y @ Ainv.t()
        
        return torch.exp(
                    torch.logsumexp(
                                torch.stack(tuple(log_cholesky_multivariate_normal(Q_Y, Q_X[i], A) for i in range(N))),
                                dim=0)) / N
    
    def log_likelihood(self, X):
        
        N, D = X.shape
        A = torch.cholesky(self.sigma)
        Ainv = torch.inverse(A)
        
        Q = X @ Ainv.t()
        
        return torch.sum(torch.logsumexp(
                    torch.stack(tuple(log_cholesky_multivariate_normal(Q[i], Q, A) - np.log(N) for i in range(N))),
                    dim=0))
    
    def save_kde(self, fname):
        
        torch.save({'sigma': self.sigma,
                    'X'    : self.X}, fname)

def load_kde(fname):
    
    d = torch.load(fname)
    kde = KDE(d['X'], None)
    kde.sigma = d['sigma']
    
    return kde

def log_cholesky_multivariate_normal(Q_test, Q_train, A):
    log_detA = torch.log(A.diagonal()).sum()
    M = A.shape[0]
    
    log_coeff = -M / 2 * np.log(2 * np.pi) - log_detA
    exp = -1 / 2 * (pairwise_dot(Q_test, Q_test) + pairwise_dot(Q_train, Q_train) - 2 * pairwise_dot(Q_test, Q_train))
    
    return log_coeff + exp

def pairwise_dot(X, Y):
    # pairwise dot product over row vectors of matrices, 
    # promotes vectors to matrices and broadcasts over singleton dimension
    
    X = atleast_2d(X)
    Y = atleast_2d(Y)
    
    if X.shape[0] == 1 and Y.shape[0] != 1:
        X = X.expand_as(Y)
    elif X.shape[0] != 1 and Y.shape[0] == 1:
        Y = Y.expand_as(X)
    
    return torch.einsum('ij,ij->i', X, Y)

def plot_kde(kde):
    
    Xn = kde.X.cpu().numpy()
    x, y = Xn[:, 0], Xn[:, 1]
    
    xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j,
             y.min():y.max():y.size ** 0.5 * 1j]
    
    Y = torch.from_numpy(np.vstack([xi.flatten(), yi.flatten()]).T).float().to(kde.X.device)
    zi = kde.density(Y).cpu().numpy()
    
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
        likelihoods.append(kde.log_likelihood(kde.X).item())
        print(iteration, likelihoods[-1])

        if kde.X.shape[1] == 2 and iteration % 3 == 0:
            plot_kde(kde)
    
    plot_training_progress(likelihoods)
