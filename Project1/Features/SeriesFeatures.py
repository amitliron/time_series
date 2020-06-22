import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time
from numpy import array
from numpy import hstack

import random

def pad(l, content, width):
    l = list(l)
    l.extend([content] * (width - len(l)))
    return np.array(l)

def normalize(data):
    return (data - min(data)) / (max(data) - min(data))

def split_sequence(sequence,X,y, n_steps,gt, id):
    if len(sequence) < n_steps:
        sequence = pad(sequence, 0, n_steps)

    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix],gt
        seq_x = np.insert(seq_x, 0, id)        # amitli
        X.append(seq_x)
        y.append(seq_y)



    return X, y

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


# # split a multivariate sequence into samples
# def split_sequences_multivariate(sequences, n_steps_in, n_steps_out):
#     X, y = list(), list()
#     for i in range(len(sequences)):
#         # find the end of this pattern
#         end_ix = i + n_steps_in
#         # check if we are beyond the dataset
#         if end_ix > len(sequences) - n_steps_out:
#             break
#         # gather input and output parts of the pattern
#         if n_steps_out > 0:
#             seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix: end_ix + n_steps_out, -1]
#         else:
#             seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1 + n_steps_out: end_ix + n_steps_out, -1]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)


def BuildDataSetForTimeSeries_Multivariate (ds, steps, bNormalize):
    start_time = time.time()
    print('Start build DataSet For TimeSeries Multivariate ....')

    uniques = ds['Id'].unique()
    X = list()
    y = list()
    ID = list()
    for id in uniques:
        f = ds.loc[ds['Id'] == id].reset_index()
        f = f.sort_values(by=['StartTime'])
        diff_array = []
        width1_array = []
        width2_array = []
        size = len(f)
        # print('size:',size)
        for i in range(1, size):
            diff = f['StartTime'][i] - f['StartTime'][i - 1]
            width1 = f['EndTime'][i - 1] - f['StartTime'][i - 1]
            width2 = f['EndTime'][i] - f['StartTime'][i]
            diff_array = np.append(diff_array, diff)
            width1_array = np.append(width1_array,width1)
            width2_array = np.append(width2_array,width2)

        # normalize data ----
        if bNormalize:
            diff_array = normalize(diff_array)
            width1_array = normalize(width1_array)
            width2_array = normalize(width2_array)

        # convert to [rows, columns] structure
        diff_array = diff_array.reshape((len(diff_array), 1))
        width1_array = width1_array.reshape((len(width1_array), 1))
        width2_array = width2_array.reshape((len(width2_array), 1))

        # horizontally stack columns
        dataset = hstack((diff_array, width1_array, width2_array))

        split_sequence(dataset, X, y, steps, f['GT'][size - 1])

    # --- return [samples, timesteps]
    print('End build DataSet For TimeSeries Multivariate: ' + 'seconds= ' + str("%.3f" % (time.time() - start_time)))
    return np.array(X), np.array(y)


def split_groups(full_x_test,  y_predict):

    class_0 = y_predict[:, 0]
    class_1 = y_predict[:, 1]

    df = pd.DataFrame({'ID': full_x_test[:, 0]})
    df['GT_0'] = class_0
    df['GT_1'] = class_1

    df_result = df.groupby('ID').mean()
    class_gt_0 = df_result['GT_0'].to_numpy()
    class_gt_1 = df_result['GT_1'].to_numpy()

    return class_gt_0, class_gt_1

def get_kolmogorov_smirnov_score(full_x_test,  y_predict):
    #
    #   input:
    #      - full_x_test (we take the first ID column)
    #      - y_predict [:,2]
    #

    #
    #   process in general:
    #   1. each ID has multiple rows (multiple windows)
    #   2. the prediction is flot [0..1]
    #   3. we get the mean of the GT prediction
    #   4. we split the values from (3) to series of GT=1 and GT=0
    #   5. plot and compare
    #

    class_0 = y_predict[:,0]
    class_1 = y_predict[:,1]

    df = pd.DataFrame({'ID': full_x_test[:, 0]})
    df['GT_0'] = class_0
    df['GT_1'] = class_1

    df_result = df.groupby('ID').mean()
    class_gt_0 = df_result['GT_0'].to_numpy()
    class_gt_1 = df_result['GT_1'].to_numpy()

    #print(df_result.head(20))

    #
    #   plot for debug only
    #
    # import matplotlib.pyplot as plt
    # df_result['GT_0'].hist(bins=10)
    # plt.show()
    # df_result['GT_1'].hist(bins=10)
    # plt.show()

    from scipy import stats
    ks_stat, p_value = stats.ks_2samp(class_gt_0, class_gt_1)

    # higher values of p_value is good
    # smaller ks_stat values is good
    return ks_stat


def train_test_split_by_IDs(X, y):
    #
    #   input:
    #           X, y - ndarray
    #
    #           X -  Id, time series values:
    #                   column  0       - ID
    #                   columns 1:end   - time series values
    #           y - target (encode as category)
    #
    #   output:
    #       X_train, X_test, y_train, y_test
    #           - encoded as ndarray
    #           - ouput contains ID as first column (as the input contains it)
    #

    #
    #   create data frame from X,y (for simplicty)
    #
    input_df = pd.DataFrame({'Id':X[:,0]})
    input_df['series'] = X[:,1:].tolist()
    gts_list = []
    for i in range(len(y)):
        if y[i][0] == 1:
            gts_list.append(0)
        else:
            gts_list.append(1)
    input_df['GT'] = gts_list


    #
    # create data frame table statistics:
    #
    #       [Id, series, GT]
    #
    statisics_ds = pd.DataFrame()
    statisics_ds['Id'] = input_df['Id'].unique()
    statisics_ds['count'] = input_df.groupby('Id')['GT'].count()
    statisics_ds['GT'] = (input_df.groupby('Id')['GT'].sum() / input_df.groupby('Id')['GT'].count())

    #
    #   get list of id's with GT = 0 or 1
    #
    list_ids_with_gt_0 = list(statisics_ds[statisics_ds['GT'] == 0]['Id'])
    list_ids_with_gt_1 = list(statisics_ds[statisics_ds['GT'] == 1]['Id'])

    #
    #   split the Id's to train and test
    #
    SPLIT_FACTOR = 0.2
    train_ids_with_gt_0 = random.sample(list_ids_with_gt_0, int(len(list_ids_with_gt_0) * (1 - SPLIT_FACTOR)))
    test_ids_with_gt_0 = np.setdiff1d(list_ids_with_gt_0, train_ids_with_gt_0).tolist()

    train_ids_with_gt_1 = random.sample(list_ids_with_gt_1, int(len(list_ids_with_gt_1) * (1 - SPLIT_FACTOR)))
    test_ids_with_gt_1 = np.setdiff1d(list_ids_with_gt_1, train_ids_with_gt_1).tolist()

    X_train_Id_list = train_ids_with_gt_0 + train_ids_with_gt_1
    X_test_Id_list = test_ids_with_gt_0 + test_ids_with_gt_1

    #
    #   prepare X_train, X_test
    #
    X_train = input_df.loc[input_df['Id'].isin(X_train_Id_list)]
    X_test = input_df.loc[input_df['Id'].isin(X_test_Id_list)]


    #
    #   prepare y_train, y_test
    #
    y_train_values = X_train['GT']
    y_test_values = X_test['GT']

    y_train = np.zeros((y_train_values.shape[0], 2))
    y_test = np.zeros((y_test_values.shape[0], 2))

    for i in range(len(y_train)):
        if y_train_values.iloc[i] == 0:
            y_train[i][0] = 1
        else:
            y_train[i][1] = 1

    for i in range(len(y_test)):
        if y_test_values.iloc[i] == 0:
            y_test[i][0] = 1
        else:
            y_test[i][1] = 1

    #X_train
    #   remove 'GT' column from X_train, X_test
    #
    X_train.drop(['GT'], axis=1, inplace=True)
    X_test.drop(['GT'], axis=1, inplace=True)

    # convert to ndarray
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    X_train = np.hstack((X_train[:, 0].reshape(-1, 1), np.array(X_train[:, 1].tolist())))
    X_test = np.hstack((X_test[:, 0].reshape(-1, 1), np.array(X_test[:, 1].tolist())))

    #
    #   return values
    #
    return X_train, X_test, y_train, y_test

# def train_test_split_2(df, X, y):
#
#     SPLIT_FACTOR = 0.2
#
#     #
#     # table statistics:
#     #
#     #       [Id, Count, GT]
#     #
#     statisics_ds = pd.DataFrame()
#     statisics_ds['Id'] = df['Id'].unique()
#     statisics_ds['count'] = df.groupby('Id')['GT'].count()
#     statisics_ds['GT'] = (df.groupby('Id')['GT'].sum() / df.groupby('Id')['GT'].count())
#
#     #
#     #   get list of id's with GT = 0 or 1
#     #
#     list_ids_with_gt_0 = list(statisics_ds[statisics_ds['GT'] == 0]['Id'])
#     list_ids_with_gt_1 = list(statisics_ds[statisics_ds['GT'] == 1]['Id'])
#
#     #
#     #   split the Id's to train and test
#     #
#     train_ids_with_gt_0 = random.sample(list_ids_with_gt_0, int(len(list_ids_with_gt_0)*(1-SPLIT_FACTOR)))
#     test_ids_with_gt_0 = np.setdiff1d(list_ids_with_gt_0, train_ids_with_gt_0)
#
#     train_ids_with_gt_1 = random.sample(list_ids_with_gt_1, int(len(list_ids_with_gt_1) * (1 - SPLIT_FACTOR)))
#     test_ids_with_gt_1 = np.setdiff1d(list_ids_with_gt_1, train_ids_with_gt_1)
#
#     X_train_Id_list = train_ids_with_gt_0 + train_ids_with_gt_1
#     X_test_Id_list = test_ids_with_gt_0 + test_ids_with_gt_1
#
#     #
#     #   prepare X_train, X_test
#     #
#     X_train = df.loc[df['Id'].isin(X_train_Id_list)]
#     X_test = df.loc[df['Id'].isin(X_test_Id_list)]
#
#     #
#     #   prepare y_train, y_test
#     #
#     y_train_values = X_train['GT']
#     y_test_values = X_test['GT']
#
#     y_train = np.zeros((y_train_values.shape[0], 2))
#     y_test = np.zeros((y_test_values.shape[0], 2))
#
#     for i in range(len(y_train)):
#         if y_train_values[i] == 0:
#             y_train[i][0] = 1
#         else:
#             y_train[i][1] = 1
#
#     for i in range(len(y_test)):
#         if y_test_values[i] == 0:
#             y_test[i][0] = 1
#         else:
#             y_test[i][1] = 1
#
#     #
#     #   remove 'GT' column from X_train, X_test
#     #
#     X_train.drop(['GT'], axis=1)
#     X_test.drop(['GT'], axis=1)
#
#     X_train = X_train.to_numpy()
#     X_test = X_test.to_numpy()
#
#     return X_train, X_test, y_train, y_test

def split_series_to_id_and_series(X_train, X_test):
    REMOVE_COL = 1
    new_x_train = np.delete(X_train, 0, REMOVE_COL)
    new_x_test = np.delete(X_test, 0, REMOVE_COL)
    return new_x_train, new_x_test

def BuildDataSetForTimeSeries_Univariate(ds, steps, varName, bNormalize):
    start_time = time.time()
    print('Start build DataSet For TimeSeries Univariate ....')

    uniques = ds['Id'].unique()

    X = list()
    y = list()
    ID = list()
    for id in uniques:
        f = ds.loc[ds['Id'] == id].reset_index()
        f = f.sort_values(by=['StartTime'])
        var_array = []
        size = len(f)
        for i in range(1, size):
            if varName == 'diff':
                v = f['StartTime'][i] - f['StartTime'][i - 1]
            elif varName == 'width1':
                v = f['EndTime'][i - 1] - f['StartTime'][i - 1]
            else:
                v = f['EndTime'][i] - f['StartTime'][i]

            var_array = np.append(var_array,v)

        # normalize data ----
        if bNormalize:
            var_array = normalize(var_array)

        split_sequence(var_array, X, y, steps, f['GT'][size - 1], id)

    # --- return [samples, timesteps]
    print('End build DataSet For TimeSeries Univariate: ' + 'seconds= ' + str("%.3f" % (time.time() - start_time)))
    return np.array(X), np.array(y)

