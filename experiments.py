from scipy.io import loadmat

from cv import KFold, Holdout
from em4kde_smart import KDE, train

X = loadmat('clusterdata2d.mat')['data']

N, D = X.shape
kde = KDE(KFold(X, 4), D)

train(kde, X, 30)