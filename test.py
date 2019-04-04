import numpy as np
from scipy.stats import multivariate_normal

def cholesky_multivariate_normal(Q_test, q_train, A):
    detA = A.diagonal().prod()
    M = A.shape[0]
    
    coeff = 1 / (2 * np.pi ** (M / 2) * detA)
    exp = -1 / 2 * (np.sum(Q_test ** 2, axis=1) + q_train.T @ q_train - 2 * Q_test @ q_train)
    
    return coeff * np.exp(exp)

# def normal(X_test, x_train, sigma):
#     N_test, M = X_test.shape
#     siginv = np.linalg.inv(sigma)
#     
#     sign, logdet = np.linalg.slogdet(2 * np.pi * sigma)
#     coeff = 1 / ((sign * np.exp(logdet)) ** (1 / 2))
#     exp = [-1 / 2 * (X_test[i] - x_train).T @ siginv @ (X_test[i] - x_train) for i in range(N_test)]
#     
#     return coeff * np.exp(exp)

import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped

scipy_times = []
# naive_times = []
cholesky_times = []

repetitions = 10
dims = 250

for M in range(1, dims):
    
    print("Dimension", M)
    
    while True:
        
        B = np.random.rand(M, M)
        sigma = B.T @ B
        
        A = np.linalg.cholesky(sigma)
        Ainv = np.linalg.inv(A)
        
        X_test = np.random.rand(10000, M) ** 2
        x_train = np.random.rand(M) ** 2
        
        Q_test = X_test @ Ainv.T
        q_train = x_train @ Ainv.T
        
        try:
            
            scipy_t = 1000 * timeit.timeit(wrapper(multivariate_normal.pdf, X_test, mean=x_train, cov=sigma),
                                           number=repetitions)
            # naive_t = 1000 * timeit.timeit(wrapper(normal, X_test, x_train, sigma), number=repetitions)
            cholesky_t = 1000 * timeit.timeit(wrapper(cholesky_multivariate_normal, Q_test, q_train, A),
                                              number=repetitions)
            
            scipy_times.append(scipy_t)
            # naive_times.append(naive_t)
            cholesky_times.append(cholesky_t)
            
            break
        
        except Exception as e:
            
            print("Got error", e)

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))

plt.plot(range(1, dims), scipy_times)
# plt.plot(range(1, dims), naive_times)
plt.plot(range(1, dims), cholesky_times)

plt.title("Normal distribution timings")
# plt.legend(["SciPy implementation", "Naïve implementation", "Cholesky implementation"])
plt.legend(["SciPy implementation", "Cholesky implementation"])
plt.xlabel("Data dimensions")
plt.ylabel("Time (milliseconds)")

plt.show()
