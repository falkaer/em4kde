import numpy as np
import torch
from scipy.io import loadmat

from cv import KFold, Holdout
from em4kde import KDE, train, load_kde

X = loadmat('clusterdata2d.mat')['data']
#y = loadmat('weather.mat')['TMPMAX']
#y = y[~np.isnan(y)]
#X = np.stack((np.arange(len(y)), y), axis=1)[:2000]

N, D = X.shape

# kde = load_kde('kde.npz')

# kde.strategy = KFold(N, 10)
kde = KDE(X, KFold(N, 10))
train(kde, 25)

# kde.save_kde('kde.npz')

# kde.strategy = KFold(N, N)
# train(kde, X, 10)
