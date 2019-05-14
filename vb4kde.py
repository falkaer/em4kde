import os
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

import torch
import numpy as np

def cov(X):
    N, _ = X.shape
    
    x_bar = X.mean(dim=0)
    diff = X - x_bar
    
    return diff.t() @ diff / (N - 1)

class KDE:
    
    def __init__(self, X, strategy):
        self.strategy = strategy
        self.X = X
        
        N, _ = X.shape
        
        self.W_0_inv, nu_0 = self.init_W_nu()
        self.W, self.nu = torch.inverse(self.W_0_inv), N + nu_0
    
    def init_W_nu(self):
        
        nu = 0.1
        return cov(self.X) * nu, nu  # TODO: maybe inverse here
    
    def e_step(self, X_train, X_test):
        N_train, D = X_train.shape
        N_test, _ = X_test.shape
        
        log_rho = torch.empty(N_train, N_test, device=self.X.device)
        
        for i in range(N_train):
            
            diff = X_test - X_train[i]
            
            # pairwise dot product over the rows of diff
            log_rho[i] = -self.nu / 2 * torch.einsum('ij,ij->i', diff @ self.W, diff)
        
        return log_rho
    
    def m_step(self, log_rho):
        
        N, D = self.X.shape
        W_inv = self.W_0_inv.clone()
        mask = ~torch.isnan(log_rho)
        
        for i in range(N):
            diff = self.X[mask[i]] - self.X[i]
            
            log_rho_i = log_rho[i, mask[i]]
            log_rho_i_sum = torch.logsumexp(log_rho[i, mask[i]], dim=0)
            rnm = torch.exp(log_rho_i - log_rho_i_sum)
            
            W_inv += (rnm[None].t() * diff).t() @ diff
        
        return torch.inverse(W_inv)
    
    def step(self):
        
        N, D = self.X.shape
        
        splits = self.strategy.get_splits()
        log_rho = torch.full((N, N), np.nan, device=self.X.device)
        
        for train_idx, test_idx in splits:
            X_train = self.X[train_idx]
            X_test = self.X[test_idx]
            
            log_rho[train_idx[:, None], test_idx] = self.e_step(X_train, X_test)
        
        self.W = self.m_step(log_rho)
    
    def density(self, Y):
        
        from torch.distributions import multivariate_normal
        
        N, D = self.X.shape
        
        # invert expectation of lambda to get sigma
        sigma = torch.inverse(self.nu * self.W)
        
        log_densities = torch.stack(
                    tuple(torch.distributions.MultivariateNormal(self.X[i], sigma).log_prob(Y) for i in range(N)),
                    dim=0)
        log_density = torch.logsumexp(log_densities, dim=0) - np.log(N)
        
        return torch.exp(log_density)
    
    def save_kde(self, fname):
        
        torch.save({'W' : self.W,
                    'nu': self.nu,
                    'X' : self.X}, fname)
    
def load_kde(fname):
    d = torch.load(fname, map_location='cpu')
    kde = KDE(d['X'], None)
    kde.W = d['W']
    kde.nu = d['nu']
    
    return kde

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

def train(kde, iterations):
    
    for iteration in range(iterations):
        
        kde.step()
        print(iteration)
        
        if kde.X.shape[1] == 2 and iteration % 3 == 0:
            plot_kde(kde)
