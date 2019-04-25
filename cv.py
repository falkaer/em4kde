from abc import abstractmethod, ABC
import numpy as np

class Strategy(ABC):
    @abstractmethod
    def get_splits(self):
        pass
    
    @abstractmethod
    def train_len(self):
        pass

def exclude_mask(N, r):
    mask = np.ones(N, dtype=bool)
    mask[r] = False
    
    return mask

class Holdout(Strategy):
    def __init__(self, N, p):
        import math

        split = math.floor(N * p)
        
        idx = np.arange(N)
        idx = np.random.permutation(idx)
        
        self.train_idx = idx[:split]
        self.test_idx = idx[split:]
        
    def get_splits(self):
        return [(self.train_idx, self.test_idx)]
    
    def train_len(self):
        return len(self.train_idx)

class KFold(Strategy):
    def __init__(self, N, k):
    
        self.N = N
        self.k = k
        
    def get_splits(self):
    
        idx = np.arange(self.N)
    
        if self.k != self.N:
            idx = np.random.permutation(idx)
    
        s = self.N // self.k
        folds = np.full(self.k, s, dtype=np.int)
        folds[:self.N % self.k] += 1  # distribute surplus observations across first few folds
    
        splits = []
        current = 0
    
        for fold_size in folds:
            start, stop = current, current + fold_size
        
            test_idx = idx[start:stop]
            splits.append((idx[exclude_mask(self.N, test_idx)], test_idx))
        
            current = stop
        
        return splits
    
    def train_len(self):
        return self.N - self.N // self.k