from abc import abstractmethod, ABC
import numpy as np
import math

def exclude_mask(N, r):
    mask = np.ones(N, dtype=bool)
    mask[r] = False
    
    return mask

class Strategy(ABC):
    @abstractmethod
    def get_splits(self):
        pass

class Holdout(Strategy):
    def __init__(self, N, p):

        self.N = N
        self.p = p
        
    def get_splits(self):
        split = math.floor(self.N * self.p)

        idx = np.arange(self.N)
        idx = np.random.permutation(idx)

        train_idx = idx[:split]
        test_idx = idx[split:]
        return [(train_idx, test_idx)]

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
            splits.append((idx[exclude_mask(self.N, range(start,stop))], test_idx))
        
            current = stop
        
        return splits
