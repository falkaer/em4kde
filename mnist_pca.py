import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

from torchvision.datasets import MNIST
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_data = MNIST('mnist', train=True, download=True)
test_data = MNIST('mnist', train=False, download=True)
X = torch.cat((train_data.data.view(train_data.data.shape[0], -1),
               test_data.data.view(test_data.data.shape[0], -1)), dim=0)

print(X.shape)

X = X.numpy().astype(np.float64)

_, D = X.shape

pca = PCA(n_components=D, whiten=True)
pca.fit(X)

cutoff_var = np.sum(pca.explained_variance_ratio_) * 0.95
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
cutoff_idx = np.arange(D)[cumsum_var > cutoff_var][0]

print(cumsum_var[cutoff_idx])
print(cutoff_var)
print(cutoff_idx)

offset = 20
xs = range(1, cutoff_idx + offset + 1)
ys = pca.explained_variance_ratio_[:cutoff_idx + offset]
d = np.zeros(cutoff_idx + offset)

plt.figure(figsize=(8, 6))
plt.fill_between(xs, ys, where=ys >= d, interpolate=True, alpha=0.2)
plt.plot(xs, ys)
plt.axvline(x=cutoff_idx + 1, c='orange')

sns.despine(left=True, bottom=True)

plt.xlabel('Dimensionality of PCA reduction', fontsize=13)
plt.ylabel('Explained variance', fontsize=13)
plt.title('MNIST choice of PCA components', fontsize=14)
plt.legend(['Explained variance', '95th percentile'])

plt.show()

import torch

sc1 = StandardScaler()
pca = PCA(n_components=cutoff_idx + 1, whiten=True)
sc2 = StandardScaler()

X_reduced = sc2.fit_transform(pca.fit_transform(sc1.fit_transform(X)))

torch.save({'X_reduced': torch.from_numpy(X_reduced),
            'sc1'      : sc1,
            'pca'      : pca,
            'sc2'      : sc2},
           'mnist_pca.pt')

# np.save('mnist_pca.npy', X_reduced)
