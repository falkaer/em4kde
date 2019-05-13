import numpy as np
import torch

import matplotlib.pyplot as plt

np.seterr(all='warn')

X_reduced = np.load('mnist_pca.npy')
print(X_reduced[0])

# from em4kde_torch import load_kde
# 
# kde_paths = ['kde_0.pt']
# 
# for fname in kde_paths:
# 
#     kde = load_kde(fname)
# 
# 
