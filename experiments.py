from scipy.io import loadmat

from cv import KFold
from em4kde import KDE, train

X = loadmat('clusterdata2d.mat')['data']

N, D = X.shape
kde = KDE(KFold(X, 4), D)

train(kde, X, 100)