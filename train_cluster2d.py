import numpy as np
import torch
from scipy.io import loadmat

np.seterr(all='warn')
from cv import KFold
import em4kde_torch as kde
import vb4kde as kde

# X = loadmat('clusterdata2d.mat')['data']
y = loadmat('weather.mat')['TMPMAX']
y = y[~np.isnan(y)]
X = np.stack((np.arange(len(y)), y), axis=1)
# np.random.shuffle(X)
X = X[:2000]
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

N, D = X.shape
X = torch.from_numpy(X).float().cuda()

# kde = load_kde('kde.npz')
# kde.strategy = KFold(N, 10)

model = kde.KDE(X, KFold(N, 3))
kde.train(model, 50)

# kde.save_kde('kde.pt')

# kde.strategy = KFold(N, N)
# train(kde, X, 10)
