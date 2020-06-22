import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time
from numpy import hstack
from numpy import array


# split a univariate sequence into samples
def split_sequence_univariate(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        # check if we are beyond the sequence
        if end_ix > len(sequence) - n_steps_out:
            break
        # gather input and output parts of the pattern
        if n_steps_out > 0:
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix + n_steps_out]
        else:
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix - 1:end_ix]

        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# split a multivariate sequence into samples
def split_sequences_multivariate(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        # check if we are beyond the dataset
        if end_ix > len(sequences) - n_steps_out:
            break
        # gather input and output parts of the pattern
        if n_steps_out > 0:
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix: end_ix + n_steps_out, -1]
        else:
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1 + n_steps_out: end_ix + n_steps_out, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)





######## univariate ###############
F1 = np.arange(10)
X, y = split_sequence_univariate(F1, 3, 2)

######## multivariate ###############
# ds = pd.DataFrame()
# ds['F1'] = np.arange(10)
# ds['F2'] = np.arange(10) * 10
# ds['Y'] = np.arange(10) * 100
# ds_array = ds.values
# X, y = split_sequences_multivariate(ds_array, 3, 4)

print('--- X ---')
for i in range(len(X)):
    print(X[i])

print('--- y ---')
for i in range(len(y)):
    print(y[i])

print('End')
