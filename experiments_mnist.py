import numpy as np
import torch

np.seterr(all='warn')

from torchvision.datasets import MNIST

from cv import KFold
from em4kde_torch import KDE, train

dataset = MNIST('mnist', train=True, download=True)
X = dataset.data.view(dataset.data.shape[0], -1)
mask = (dataset.targets == 0)
X = X[mask].numpy()

from sklearn.decomposition import PCA

X = PCA(n_components=500).fit_transform(X)
N, D = X.shape
X = X.astype(np.float64)

print(N, D)

X = torch.from_numpy(X).float().cuda()
kde = KDE(X, KFold(N, 3))
train(kde, 50)

kde.save_kde('kde_0.npz')

# kde.strategy = KFold(N, N)
# train(kde, X, 10)
