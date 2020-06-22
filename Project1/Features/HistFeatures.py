import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time

def PlotRowHist(data,dist,gt):
    fig, axs = plt.subplots(3, 1)
    # axs[0].plot(np.arange(len(row)), row)
    if gt == 0:
        col = 'b'
    else:
        col = 'r'

    axs[0].plot(np.arange(len(data)), data,label='diff',color=col)
    axs[1].hist(data, 100, density=False, label='diff hist',color=col)
    axs[2].bar(np.arange(len(dist)), dist, label='norm diff hist',color=col)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.grid()
    # plt.title(str(gt))
    plt.show()

def DataDist(data,bins):
    x = np.histogram(data, bins=bins)[0]
    s = np.sum(x)
    x_w = x / s
    return x_w


def BuildDataSet(ds, bins):
    '''

    :param ds:
    :param bins:
    :return:
        dataframe: [#IDs, 3*#Bins+1]
          +1 ofr GT
    '''
    start_time = time.time()
    print('Start build bins features ....')
    # --- build ds ----
    cols = []
    for i in range(1, bins + 1):
        cols = np.append(cols, 'BinDiff_' + str(i))

    for i in range(1, bins + 1):
        cols = np.append(cols, 'BinWidth1_' + str(i))

    for i in range(1, bins + 1):
        cols = np.append(cols, 'BinWidth2_' + str(i))

    cols = np.append(cols, 'GT')

    all = []

    uniques = ds['Id'].unique()

    for id in uniques:
        f = ds.loc[ds['Id'] == id].reset_index()
        f = f.sort_values(by=['StartTime'])
        diff_array = []
        width1_array = []
        width2_array = []
        size = len(f)
        # print('size:',size)
        for i in range(1,size):
            diff = f['StartTime'][i] - f['StartTime'][i-1]
            width1 = f['EndTime'][i-1] - f['StartTime'][i-1]
            width2 = f['EndTime'][i] - f['StartTime'][i]
            diff_array = np.append(diff_array, diff)
            width1_array = np.append(width1_array, width1)
            width2_array = np.append(width2_array, width2)


        BinsDiff = DataDist(diff_array, bins)
        Binswidth1 = DataDist(width1_array, bins)
        Binswidth2 = DataDist(width2_array, bins)

        # ---- Plot row -------
        # if id% 100 == 0:
        #     PlotRowHist(diff_array,BinsDiff, f['GT'][size - 1])

        row = []
        row = np.append(row, BinsDiff)
        row = np.append(row, Binswidth1)
        row = np.append(row, Binswidth2)
        row = np.append(row, f['GT'][size-1])
        all.append(row)

    BinsFeatures_ds = pd.DataFrame(all, columns=cols)
    print('End build bins features: ' + 'seconds= ' + str("%.3f" % (time.time() - start_time)))

    print("")
    print(BinsFeatures_ds.head(4))
    print("")

    return BinsFeatures_ds

def PlotFeaturesHists(ds):
    cols = list(ds.columns)
    for c in cols:
        if c != 'GT':
            gt_0 = ds[c][ds['GT'] == 0]
            gt_1 = ds[c][ds['GT'] == 1]

            plt.hist(gt_0, 100, density=True, label=0,color='b', alpha=1.0)
            plt.hist(gt_1, 100, density=True, label=1, color='r', alpha=1.0)

            # values1 = np.array((df.loc[df[gtColName] == y1]).iloc[:, f])
            # values2 = np.array((df.loc[df[gtColName] == y2]).iloc[:, f])
            res = stats.ks_2samp(gt_0, gt_1)

            title = c + ' (ks=' +  str('%.4f' % res[0]) + ')'
            plt.title(title)
            plt.legend()
            plt.grid()

            plt.show()
            plt.clf()

    print(cols)
