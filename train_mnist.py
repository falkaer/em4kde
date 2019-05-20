import numpy as np
import torch

np.seterr(all='warn')

from torchvision.datasets import MNIST

from cv import KFold
import vb4kde as kde
# import em4kde_torch as kde

dataset = MNIST('mnist', train=True, download=True)
mask = torch.cat(((dataset.targets == 8), torch.zeros(10000, dtype=torch.uint8)))

pca_data = torch.load('mnist_pca_90.pt')
X_reduced = pca_data['X_reduced'][mask].float().cuda()

N, D = X_reduced.shape
print(N, D)

model = kde.KDE(X_reduced, KFold(N, 10))
kde.train(model, 50)

model.save_kde('vb_em_8_90.pt')
