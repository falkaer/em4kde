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
        
        ln_rho = torch.empty(N_train, N_test, device=self.X.device)
        
        for i in range(N_train):
            diff = X_test - X_train[i]
            ln_rho[i] = -self.nu / 2 * diff.t() @ self.W @ diff
        
        logsumexp = torch.logsumexp(ln_rho, dim=0)
        
        return torch.exp(ln_rho - logsumexp)
    
    def m_step(self, rnm):
        
        N, D = self.X.shape
        W_inv = self.W_0_inv.clone()
        mask = ~torch.isnan(rnm)
        
        for i in range(N):
            diff = self.X[mask[i]] - self.X[i]
            rnm_sum = rnm[i, mask[i]].sum()
            
            W_inv += rnm_sum * diff @ diff.t()
        
        return torch.inverse(W_inv)
    
    def step(self):
        
        N, D = self.X.shape
        
        splits = self.strategy.get_splits()
        gamma = torch.full((N, N), np.nan, device=self.X.device)
        
        for train_idx, test_idx in splits:
            
            X_train = self.X[train_idx]
            X_test = self.X[test_idx]

            gamma[train_idx[:, None], test_idx] = self.e_step(X_train, X_test)
        
        self.W = self.m_step(gamma)
        