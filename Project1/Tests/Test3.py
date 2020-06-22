import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time


def DataDist(data,bins):
    x = np.histogram(data, bins=bins)[0]
    s = np.sum(x)
    x_w = x / s
    return x_w

v1 = np.array([70,75,76,75,80,70,120,121,120,121,123,120,70,60])
v2 = v1 - 40

v1_dist = DataDist(v1,50)
v2_dist = DataDist(v2,50)

# a = np.reshape(v1_dist, v1_dist.shape[0])
# b = np.array(v1_dist)
# z = [1,2,3]
# s = list(v1_dist)

fig, axs = plt.subplots(2, 2)

axs[0, 0].hist(v1, 100, density=False, alpha=1.0)
axs[0, 1].hist(v2, 100, density=False, alpha=1.0)

# axs[1, 0].hist(v1_dist, 50, density=False, alpha=1.0)
# axs[1, 1].hist(v2_dist, 50, density=False, alpha=1.0)

axs[1, 0].bar(np.arange(len(v1_dist)), v1_dist)
axs[1, 1].bar(np.arange(len(v2_dist)), v2_dist)

plt.show()