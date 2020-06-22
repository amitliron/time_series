import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time
from numpy import hstack

from Features import SeriesFeatures as sf

F1_f1 = np.arange(9)+10
F1_f2 = np.arange(9)+100
F2_f1 = np.arange(7)+40
F2_f2 = np.arange(7)+400
print('F1 Feature1',F1_f1)
print('F1 Feature2',F1_f2)
print('F2 Feature1',F2_f1)
print('F2 Feature2',F2_f2)

# convert to [rows, columns] structure
F1_f1_seq = F1_f1.reshape((len(F1_f1), 1))
F1_f2_seq = F1_f2.reshape((len(F1_f2), 1))
F2_f1_seq = F2_f1.reshape((len(F2_f1), 1))
F2_f2_seq = F2_f2.reshape((len(F2_f2), 1))

dataset_F1 = hstack((F1_f1_seq, F1_f2_seq))
dataset_F2 = hstack((F2_f1_seq, F2_f2_seq))
print(dataset_F1)
print('-------------')
print(dataset_F2)

print('######### F1 ###########')
X = list()
y = list()
n_steps = 3
sf.split_sequence(dataset_F1,X,y,n_steps,0)
X = np.array(X)

for i in range(len(X)):
	print(X[i])

print('######### F2 ###########')
X = list()
y = list()
n_steps = 3
sf.split_sequence(dataset_F2,X,y,n_steps,1)
X = np.array(X)

for i in range(len(X)):
	print(X[i])