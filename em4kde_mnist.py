import numpy as np
import torch
from torchvision.datasets import MNIST
from scipy.io import loadmat

from cv import KFold, Holdout
from em4kde_smart import KDE, train

# X = loadmat('clusterdata2d.mat')['data']
#y = loadmat('weather.mat')['TMPMAX']
#y = y[~np.isnan(y)]
#X = np.stack((np.arange(len(y)), y), axis=1)[:2000]

dataset = MNIST('mnist', train=True, download=True)
N, _, _ = dataset.train_data.shape
X = dataset.train_data.view(N, -1)[0:1000].numpy()
N, _ = X.shape

kde = KDE(X, KFold(N, 10))
train(kde, X, 100)

# kde.strategy = KFold(N, N)
# train(kde, X, 10)
