import numpy as np
from scipy.io import loadmat

from cv import KFold, Holdout
from em4kde_smart import KDE, train

X = loadmat('clusterdata2d.mat')['data']
#y = loadmat('weather.mat')['TMPMAX']
#y = y[~np.isnan(y)]
#X = np.stack((np.arange(len(y)), y), axis=1)[:2000]

print(X.shape)

N, D = X.shape

kde = KDE(X, Holdout(N, 0.5))

train(kde, X, 100)
