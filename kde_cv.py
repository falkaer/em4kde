from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class SklearnKde:
    
    def __init__(self, bandwidths, X, folds):
        self.bandwidths = bandwidths
        self.X = X
        self.grid = None
        self.folds = folds
    
    def train_CV(self):
        params = {'bandwidth': self.bandwidths}
        
        self.grid = GridSearchCV(
                    KernelDensity(kernel='gaussian'),
                    params,
                    cv=KFold(self.folds, shuffle=True))
        
        self.grid.fit(self.X)
    
    def result(self):
        print("best bandwidth: {0}".format(self.grid.best_estimator_.bandwidth))
        return self.grid.best_estimator_
    
    def plot(self):
        x, y = self.X[:, 0], self.X[:, 1]
        
        xi, yi = np.mgrid[x.min():x.max():x.size ** 0.5 * 1j, y.min():y.max():y.size ** 0.5 * 1j]
        
        # fit optimal kde
        bw = self.grid.best_estimator_.bandwidth
        kde_skl = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde_skl.fit(np.vstack([y, x]).T)
        
        # score_samples() returns the log-likelihood of the samples
        zi = np.exp(kde_skl.score_samples(np.vstack([yi.ravel(), xi.ravel()]).T))
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xi, yi, zi.reshape(xi.shape))
        plt.scatter(x, y, s=2, color='white')
        
        plt.gca().set_xlim(x.min(), x.max())
        plt.gca().set_ylim(y.min(), y.max())
        plt.title('sklearn kde')
        # plt.colorbar(c)
        
        plt.show()

import numpy as np
#from scipy.io import loadmat
import torch

np.seterr(all='warn')

from torchvision.datasets import MNIST

dataset = MNIST('mnist', train=True, download=True)
mask = torch.cat(((dataset.targets == 8), torch.zeros(10000, dtype=torch.uint8)))

pca_data = torch.load('mnist_pca.pt')
X = pca_data['X_reduced'][mask].numpy()
# X = loadmat('clusterdata2d.mat')['data']
#y = loadmat('weather.mat')['TMPMAX']
#y = y[~np.isnan(y)]
#X = np.stack((np.arange(len(y)), y), axis=1)[:2000]

kde = SklearnKde(np.linspace(0.0000000001, 5, 100), X, 10)
kde.train_CV()
kde.result()
#kde.plot()
