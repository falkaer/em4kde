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
        
        if k != N:
            np.random.permutation(X)

        s = N // k
        self.splits = [(X[exclude_mask(N, range(i * s, (i + 1) * s))], X[i * s:(i + 1) * s]) for i in range(k)]
    
    def get_splits(self):
        return self.splits
