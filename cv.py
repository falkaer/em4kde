from abc import abstractmethod, ABC
import numpy as np

class Strategy(ABC):
    @abstractmethod
    def get_splits(self):
        pass

def exclude_mask(N, r):
    mask = np.ones(N, dtype=bool)
    mask[r] = False
    
    return mask

class KFold(Strategy):
    def __init__(self, X, k):
        
        self.k = k
        N, D = X.shape
        
        idx = np.arange(N)
        
        if k != N:
            idx = np.random.permutation(idx)
        
        s = N // k
        folds = np.full(k, s, dtype=np.int)
        folds[:N % k] += 1 # distribute surplus observations across first few folds
        
        self.splits = []
        current = 0
        
        for fold_size in folds:
            start, stop = current, current + fold_size
            
            test_idx = idx[start:stop]
            self.splits.append((X[exclude_mask(N, test_idx)], X[test_idx]))
            
            current = stop
        
    def get_splits(self):
        return self.splits
