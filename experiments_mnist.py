import numpy as np
import torch

np.seterr(all='warn')

from torchvision.datasets import MNIST

from cv import KFold
from em4kde_torch import KDE, train

dataset = MNIST('mnist', train=True, download=True)
mask = (dataset.targets == 0)

X_reduced = np.load('mnist_pca.npy')
X_reduced = torch.from_numpy(X_reduced)[mask].float().cuda()

N, D = X_reduced.shape
print(N, D)

kde = KDE(X_reduced, KFold(N, 10))
train(kde, 100)

kde.save_kde('kde_0.pt')
