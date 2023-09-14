import numpy as np
from scipy.stats import multivariate_normal

# 3 hidden states, pi and transfer matrix
pi = [1 / 3, 1 / 3, 1 / 3]
states = [0, 1, 2]

T = [[0.4, 0.3, 0.3],
     [0.3, 0.4, 0.3],
     [0.3, 0.3, 0.4]]
T = [[0.8,0.1,0.1],
     [0.1,0.8,0.1],
     [0.1,0.1,0.8]]

T = np.array(T)

# generate observation sequence based on hidden states sequence and guassian distribution
def generator(z):
    if z == 0:
        mu = [1, 1, -1, 0]
        sigma = [[2, 0.6, -0.5, 0.8],
                 [0.6, 2, 0.7, 0.8],
                 [-0.5, 0.7, 2, 0.8],
                 [0.8, 0.8, 0.8, 1]]
        #print(np.linalg.det(np.array(sigma)))
        return multivariate_normal.rvs(mean=mu, cov=sigma)
    elif z == 1:
        mu = [-1, 1, 1, 0]
        sigma = [[2, 0.6, -0.5, 0.8],
                 [0.6, 2, 0.7, 0.8],
                 [-0.5, 0.7, 2, 0.8],
                 [0.8, 0.8, 0.8, 1]]
        return multivariate_normal.rvs(mean=mu, cov=sigma)
    else:
        mu = [1, -1, 1, 0]
        sigma = [[2, 0.6, -0.5, 0.8],
                 [0.6, 2, 0.7, 0.8],
                 [-0.5, 0.7, 2, 0.8],
                 [0.8, 0.8, 0.8, 1]]
        return multivariate_normal.rvs(mean=mu, cov=sigma)

# generate new hidden states based on the former states
def transform(z):
    return np.random.choice(np.arange(0, 3), p=T[int(z)])
