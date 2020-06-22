import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

'''
GT
    F
        Repet
            patternOredr
                Pattern
                    PatternQ
'''

gt_pattern = [[0, 1], [2, 3]]
gt_Q = [20, 20]

# --- 80% -------
# pattern_Q_mean = [15, 15, 15, 15]
# pattern_Q_std = [5, 5, 5, 5]
#
# pattern_Diff_mean = [12, 12, 16, 11]
# pattern_Diff_std = [3, 3, 3, 3]
#
# pattern_Width_mean = [7, 7, 11, 9]
# pattern_Width_std = [3, 3, 3, 3]

# --- 99% -------
pattern_Q_mean = [7, 15, 35, 55]
pattern_Q_std = [3, 3, 5, 5]

pattern_Diff_mean = [7, 12, 30, 50]
pattern_Diff_std = [3, 3, 7, 7]

pattern_Width_mean = [7, 7, 20, 25]
pattern_Width_std = [3, 3, 5, 5]

pattern_Repet_mean = [25, 25]
pattern_Repet_std = [4, 4]

permutationsDic = {}



def GetSimulatData(fileName):
    start_time = time.time()
    if os.path.isfile('./RowData/' + fileName + '.csv'):
        msg = 'load data'
        print('Start ' + msg + ' ....')
        ds = pd.read_csv('./RowData/' + fileName + '.csv')
    else:
        msg = 'Start simulate data'
        print('Start ' + msg + ' ....')
        # --- BuildPermutationsDic -----
        for x in range(0, len(gt_pattern)):
            permutationsDic[x] = list(itertools.permutations(gt_pattern[x]))

        all = []
        id_index = 0
        for gt in range(0, len(gt_pattern)):
            for f in range(0, gt_Q[gt]):
                currentTime = 0.0
                repet = (int)(np.random.normal(pattern_Repet_mean[gt], pattern_Repet_std[gt], 1))
                for r in range(0, repet):
                    patternOredr = random.choice(permutationsDic[gt])
                    for pOr in patternOredr:
                        Q = (int)(np.random.normal(pattern_Q_mean[pOr], pattern_Q_std[pOr], 1))
                        for q in range(0, Q):
                            diff = np.random.normal(pattern_Diff_mean[pOr], pattern_Diff_std[pOr], 1)[0]
                            startTime = currentTime + diff
                            endTime = startTime + np.random.normal(pattern_Width_mean[pOr], pattern_Width_std[pOr], 1)[0]
                            currentTime = startTime
                            all.append([id_index, startTime, endTime, gt])
                id_index += 1

        ds = pd.DataFrame(all, columns=['Id', 'StartTime', 'EndTime', 'GT'])
        ds.to_csv('./RowData/' + fileName + '.csv', index=False)

        print('End' + msg + ' seconds= ' + str("%.3f" % (time.time() - start_time)))
    return ds


def GetDemoData1():
    ds = pd.DataFrame()
    startDiff = []
    widthDiff = [0]
    starts = [0]
    ends = [0]
    ids = [0]
    gts = [0]
    s = 20
    for i in range(0,s):
        startDiff = np.append(startDiff,i * 100)
        widthDiff = np.append(widthDiff, i * 2 + 1)

    for i in range(1,s):
        start = starts[i-1] + startDiff[i]
        starts = np.append(starts, start)
        id = (int)(i / 5)
        if i % 5 < 3:
            gt = 0
        else:
            gt = 1

        end = starts[i] + widthDiff[i]
        ends = np.append(ends, end)
        ids = np.append(ids, id)
        gts = np.append(gts, gt)

    ds['Id'] = ids[1:]
    ds['StartTime'] = starts[1:]
    ds['EndTime'] = ends[1:]
    ds['GT'] = gts[1:]


    # ds['Id'] = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    # ds['StartTime'] = [100, 300, 600, 1000, 1500, 2100, 2800, 3600, 4500, 5500, 6600, 7800, 9100, 10500, 12000, 13600, 15300]
    # ds['EndTime'] = [3, 10, 23, 42, 67, 98, 135, 178, 227, 282, 343]
    # ds['GT'] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    return ds


def PlotDataSet(ds):
    plt.plot(ds['StartTime'][ds['GT'] == 0], ds['Id'][ds['GT'] == 0], 'bo', label='0')
    plt.plot(ds['StartTime'][ds['GT'] == 1], ds['Id'][ds['GT'] == 1], 'ro', label='1')
    plt.legend()
    plt.show()


def PlotDataSetRec(ds):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for index, row in ds.iterrows():
        x = row['StartTime']
        y = row['Id']
        w = row['EndTime'] - row['StartTime']
        h = 0.2

        if row['GT'] == 0:
            rect = plt.Rectangle((x, y), w, h, color='b', alpha=1.0)
        else:
            rect = plt.Rectangle((x, y), w, h, color='r', alpha=1.0)
        ax.add_patch(rect)

    ax.set_xlim(0, ds['EndTime'].max())
    ax.set_ylim(0, ds['Id'].max())
    plt.show()


def GetSatistic(ds):
    gt_0 = ds[ds['GT'] == 0]
    z_0 = gt_0.groupby('Id').count()

    gt_1 = ds[ds['GT'] == 1]
    z_1 = gt_1.groupby('Id').count()

    plt.hist(z_0['GT'], 100, density=False, label=0, color='b', alpha=1.0)
    plt.hist(z_1['GT'], 100, density=False, label=1, color='r', alpha=1.0)

    title = 'Event count hist'
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.show()
    plt.clf()

def PrintDs(X):
    s = len(X)
    for i in range(s):
        print(X[i])


# ds = GetSimulatData('Row99_f=40')
# PlotDataSetRec(ds)
