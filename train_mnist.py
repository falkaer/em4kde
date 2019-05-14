import numpy as np
import torch

np.seterr(all='warn')

from torchvision.datasets import MNIST

from cv import KFold
import em4kde_torch
import vb4kde

dataset = MNIST('mnist', train=True, download=True)
mask = torch.cat(((dataset.targets == 0), torch.zeros(10000, dtype=torch.uint8)))

pca_data = torch.load('mnist_pca.pt')
X_reduced = pca_data['X_reduced'][mask].float().cuda()

N, D = X_reduced.shape
print(N, D)

em_kde = em4kde_torch.KDE(X_reduced, KFold(N, 10))
em4kde_torch.train(em_kde, 20)

em_kde.save_kde('kde_em_0.pt')

vb_kde = vb4kde.KDE(X_reduced, KFold(N, 10))
vb4kde.train(vb_kde, 50)

vb_kde.save_kde('kde_vb_0.pt')