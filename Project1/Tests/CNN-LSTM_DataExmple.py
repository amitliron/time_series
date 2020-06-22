import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time

from Features import SeriesFeatures as sf

F1 = np.arange(9)+10
F2 = np.arange(7)+40
print(F1)
print(F2)


X = list()
y = list()
sf.split_sequence(F1,X,y,6,0)
X = np.array(X)

# for i in range(len(X)):
# 	print(X[i])

# --- Cnn-Lstm ---
n_features = 1
n_seq = 2
n_steps = 3
X = X.reshape((X.shape[0], n_seq, n_steps))
s = len(X)
for i in range(s):
    print(X[i])

print()

X = list()
y = list()
sf.split_sequence(F2,X,y,6,0)
X = np.array(X)

# for i in range(len(X)):
# 	print(X[i])

# --- Cnn-Lstm ---
n_features = 1
n_seq = 2
n_steps = 3
X = X.reshape((X.shape[0], n_seq, n_steps))
s = len(X)
for i in range(s):
    print(X[i])




print('hhh')
