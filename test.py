import numpy as np
from scipy.stats import multivariate_normal

def cholesky_multivariate_normal(Q_test, q_train, A):
    detA = A.diagonal().prod()
    M = A.shape[0]
    
    # coeff = (2 * np.pi) ** (- M / 2) / detA
    coeff = 1 / (2 * np.pi ** (M/2) * detA)
    exp = -1 / 2 * (np.sum(Q_test ** 2, axis=1) + q_train.T @ q_train - 2 * Q_test @ q_train)
    # print("cholesky exp:", exp)
    
    return coeff * np.exp(exp)

def normal(X_test, x_train, sigma):
    
    N_test, M = X_test.shape
    siginv = np.linalg.inv(sigma)
    
    coeff = 1 / (np.linalg.det(2 * np.pi * sigma) ** (1/2))
    exp = [-1 / 2 * (X_test[i] - x_train).T @ siginv @ (X_test[i] - x_train) for i in range(N_test)]
    # print("normal exp:", exp)
    
    return coeff * np.exp(exp)

import timeit

scipy_times = []
naive_times = []
cholesky_times = []

for M in range(100):
    
    sigma = np.random.rand(M, M)
    sigma = sigma.T @ sigma
    
    A = np.linalg.cholesky(sigma)
    Ainv = np.linalg.inv(A)
    
    X_test = np.random.rand(100, M) ** 2
    x_train = np.random.rand(M) ** 2
    
    Q_test = X_test @ Ainv.T
    q_train = x_train @ Ainv.T
    
    multivariate_normal.pdf(X_test, mean=x_train, cov=sigma)
    print(normal(X_test, x_train, sigma))
    print(cholesky_multivariate_normal(Q_test, q_train, A))
    
    print("Dimension", M)