import numpy as np
from scipy.stats import multivariate_normal

def cholesky_multivariate_normal(Q_test, Q_train, A):
    if len(Q_train.shape) == 1:
        Q_switch = Q_test
        Q_test = Q_train
        Q_train = Q_switch
    
    log_detA = np.log(A.diagonal()).sum()
    M = A.shape[0]
    
    log_coeff = -M / 2 * np.log(2 * np.pi) - log_detA
    # exp = -1 / 2 * (pairwise_dot(Q_test, Q_test) + pairwise_dot(Q_train, Q_train) - 2 * pairwise_dot(Q_test, Q_train))
    exp = -1 / 2 * (Q_test @ Q_test + np.einsum('ij,ij->i', Q_train, Q_train) - 2 * Q_train @ Q_test)
    
    return np.exp(log_coeff + exp)

import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped

scipy_times = []
cholesky_times = []

repetitions = 10
dims = 250

for M in range(1, dims):
    
    print("Dimension", M)
    
    while True:
        
        B = np.random.rand(M, M)
        sigma = B @ B.T
        
        A = np.linalg.cholesky(sigma)
        Ainv = np.linalg.inv(A)
        
        X_test = np.random.rand(10000, M) ** 2
        x_train = np.random.rand(M) ** 2
        
        Q_test = X_test @ Ainv.T
        q_train = x_train @ Ainv.T
        
        try:
            
            scipy_t = 1000 * timeit.timeit(wrapper(multivariate_normal.pdf, X_test, mean=x_train, cov=sigma),
                                           number=repetitions)
            cholesky_t = 1000 * timeit.timeit(wrapper(cholesky_multivariate_normal, Q_test, q_train, A),
                                              number=repetitions)
            
            scipy_times.append(scipy_t)
            cholesky_times.append(cholesky_t)
            
            break
        
        except Exception as e:
            
            print("Got error", e)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

plt.figure(figsize=(6, 4))

plt.plot(range(1, dims), scipy_times)
plt.plot(range(1, dims), cholesky_times)

plt.title("Normal distribution timings")
plt.legend(["SciPy implementation", "Cholesky implementation"])
plt.xlabel("Data dimensions")
plt.ylabel("Time (milliseconds)")

plt.show()
