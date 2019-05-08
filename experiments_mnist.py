from torchvision.datasets import MNIST

from cv import KFold
from em4kde import KDE, train

dataset = MNIST('mnist', train=True, download=True)
X = dataset.data.view(dataset.data.shape[0], -1)
mask = (dataset.targets == 0)
X = X[mask].numpy()

from sklearn.decomposition import PCA

X_reduced = PCA(n_components=100).fit_transform(X)
N, D = X_reduced.shape

print(N, D)

kde = KDE(X_reduced, KFold(N, 3))
train(kde, 100)

kde.save_kde('kde_0.npz')

# kde.strategy = KFold(N, N)
# train(kde, X, 10)
