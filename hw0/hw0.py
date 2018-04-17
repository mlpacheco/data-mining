import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats


np.random.seed(1234)


'''
samples = np.random.normal(loc=0.0, scale=1.0, size=(NUM_OBS,SZ))
SZ = 100
NUM_OBS = 100


# Uncomment for plotting
plt.scatter(samples[:,0], samples[:,1])
plt.show()
'''

'''
# Uncomment for empirical probabilities
samples = [s for s in samples if (abs(s) <= 2).sum() == SZ]
distances = [np.linalg.norm(s-np.zeros(SZ)) for s in samples]
inside = [d for d in distances if d <= 0.5]
print(len(inside)*1.0 / len(distances))
'''


# Uncomment for gradient ascent
'''
NUM_ITER = 1000
LR = 0.01

samples = np.random.normal(loc=0.0, scale=1.0, size=(100,))
params = np.random.rand(2)
n =  samples.shape[0] * 1.0

for i in range(NUM_ITER):

    gradient = np.asarray([
        (samples - params[0]).sum() / params[1],
        - n / (2.0 * params[1]) + \
                ((samples - params[0])**2).sum() / (2 * params[1]**2)
    ])

    params += LR * (gradient/n)

print ("estimated_params", params)
print ("obs_params", [np.mean(samples), np.var(samples)])
'''


# Assume file is correct
season_dict = {'winter': -1, 'summer': 1, 'fall': 0, 'spring': 0}

max_i = 0; max_j = 0; tuples = []
with open('matrix.csv') as f:
    for line in f:
        i, j, season = line.split(',')
        i = int(i.strip()); j = int(j.strip())
        max_i = max(max_i, i); max_j = max(max_j, j)
        tuples.append((i, j, season.strip()))

    A = np.empty((max_i + 1, max_j + 1), int)
    A[:] = np.nan
    for i,j,season in tuples:
        if season in season_dict:
            A[i,j] = season_dict[season]

print(A)
print
print(A[np.array([1, 2])[:,None], np.array([9, 10])])
