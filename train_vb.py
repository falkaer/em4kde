import numpy as np
import torch
from scipy.io import loadmat

np.seterr(all='warn')
from cv import KFold
from vb4kde import KDE, train, load_kde

# X = loadmat('clusterdata2d.mat')['data']
y = loadmat('weather.mat')['TMPMAX']
y = y[~np.isnan(y)]
X = np.stack((np.arange(len(y)), y), axis=1)[:2000]

N, D = X.shape
X = torch.from_numpy(X).float()#.cuda()

# kde = load_kde('kde.npz')
# kde.strategy = KFold(N, 10)

kde = KDE(X, KFold(N, 3))
train(kde, 20)

# kde.save_kde('kde.npz')

# kde.strategy = KFold(N, N)
# train(kde, X, 10)
