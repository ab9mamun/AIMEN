import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import external.sortutil as sortutil
import math


def add_HRCP_column(basepath):
    datadf = pd.read_csv(basepath + 'fri_data.csv')
    print(datadf.head())
    cols = datadf.columns
    for i, col in enumerate(cols):
        print(i, col)

    n = datadf.shape[0]
    apgars1 = datadf['apgars1'].to_numpy()
    ph = datadf['ph'].to_numpy()
    CP = datadf['CP'].to_numpy()
    HRCP = [0 for i in range(n)]
    for i in range(n):
        if apgars1[i] <= 3 or ph[i] <= 7.05 or CP[i] == 1:
            HRCP[i] = 1
    print(HRCP)
    datadf = datadf.assign(HRCP=HRCP)
    datadf.to_csv(basepath+'fri_data_v2.csv', index=False)


def bar_plot(column, title, basepath, col):
    plt.figure(figsize=(10,10))
    column = column[~np.isnan(column)]
    if col == 'ph':
        column = np.array(column).astype(float)
    else:
        column = np.array(column).astype(int)
    values, counts = np.unique(column, return_counts=True)

    if len(values) > 10:
        zipped = sorted(zip(values, counts), key=lambda x: x[0])
        sorted_values_original = [x for x, _ in zipped]
        sorted_counts_count = [x for _, x in zipped]

        minval = sorted_values_original[0]
        maxval = sorted_values_original[-1]
        if col == 'ph':
            minval = math.floor(minval*20)/20
            maxval = math.ceil(maxval*20)/20
            step = 0.05
            num_steps = int((maxval - minval)/step)
            diff = 0.01
        elif col == 'weightgrams':
            minval = math.floor(minval/200)*200
            maxval = math.ceil(maxval/200)*200
            step = 200
            num_steps = int((maxval - minval)/step)
        else:
            minval = math.floor(minval)
            maxval = math.ceil(maxval)
            num_steps = 10
            step = (maxval - minval)/num_steps
            if step < 1:
                step = 1
            else:
                step = math.floor(step)
            num_steps = int((maxval - minval)/step)

        sorted_values = [minval + i*step for i in range(num_steps+1)]
        sorted_counts = [0 for i in range(num_steps+1)]
        for i in range(len(sorted_values_original)):
            index = int((sorted_values_original[i] - minval)/step)
            sorted_counts[index] += sorted_counts_count[i]
        if col == 'ph':
            ytick_labels = [f'{minval:.2f} - {minval+step-diff:.2f}' for minval in sorted_values]
        elif step > 1:
            ytick_labels = [f'{minval} - {minval+step-1}' for minval in sorted_values]
        else:
            ytick_labels = [f'{minval}' for minval in sorted_values]

    else:
        zipped = sorted(zip(values, counts), key=lambda x: x[0])
        sorted_values = [x for x,_ in zipped]
        sorted_counts = [x for _,x in zipped]
        ytick_labels = [str(x) for x in sorted_values]
        #width = 0.75
    offset = max(sorted_counts)/100
    plt.barh(np.arange(len(sorted_values)), sorted_counts)
    for i in range(len(sorted_values)):
        plt.text(sorted_counts[i]+offset, i, str(sorted_counts[i]))
    plt.title(title)
    plt.ylabel(f'{col} value')
    plt.xlabel('Count')
    plt.yticks(np.arange(len(sorted_values)), ytick_labels)
    plt.savefig(basepath+'plots/'+title+'.png')

def bar_plot_combined(hrcp, column, title, basepath, col):
    plt.figure(figsize=(10, 10))
    column = column[~np.isnan(column)]
    if col == 'ph':
        column = np.array(column).astype(float)
    else:
        column = np.array(column).astype(int)
    values = np.unique(column)

    hrcp_count = [0 for i in range(len(values))]
    normal_count = [0 for i in range(len(values))]

    if len(values) > 10:

        minval = np.min(values)
        maxval = np.max(values)
        if col == 'ph':
            minval = math.floor(minval * 20) / 20
            maxval = math.ceil(maxval * 20) / 20
            step = 0.05
            num_steps = int((maxval - minval) / step)
            diff = 0.01
        elif col == 'weightgrams':
            minval = math.floor(minval / 200) * 200
            maxval = math.ceil(maxval / 200) * 200
            step = 200
            num_steps = int((maxval - minval) / step)
        else:
            minval = math.floor(minval)
            maxval = math.ceil(maxval)
            num_steps = 10
            step = (maxval - minval) / num_steps
            if step < 1:
                step = 1
            else:
                step = math.floor(step)
            num_steps = int((maxval - minval) / step)

        range_starters = [minval + i * step for i in range(num_steps)]
        normal_count = [0 for i in range(num_steps)]
        hrcp_count = [0 for i in range(num_steps)]

        for i in range(num_steps):
            ind1 = np.where(column >= minval+i*step)[0]
            ind2 = np.where(column < minval+(i+1)*step)[0]
            indices = np.intersect1d(ind1, ind2)
            hrcp_for_this_val = hrcp[indices]
            count = np.count_nonzero(hrcp_for_this_val)
            hrcp_count[i] = count
            normal_count[i] = len(hrcp_for_this_val) - count

        if col == 'ph':
            ytick_labels = [f'{minval:.2f} - {minval + step - diff:.2f}' for minval in range_starters]
        elif step > 1:
            ytick_labels = [f'{minval} - {minval + step - 1}' for minval in range_starters]
        else:
            ytick_labels = [f'{minval}' for minval in range_starters]

        ind = np.arange(num_steps)

    else:
        for i, val in enumerate(values):
            indices = np.where(column == val)[0]
            hrcp_for_this_val = hrcp[indices]
            count = np.count_nonzero(hrcp_for_this_val)
            hrcp_count[i] = count
            normal_count[i] = len(hrcp_for_this_val) - count

        ytick_labels = [str(x) for x in values]
        ind = np.arange(len(values))

    width = 0.4

    offset1 = max(normal_count) / 100
    offset2 = max(hrcp_count) / 100

    plt.barh(ind, normal_count, width)
    plt.barh(ind-width, hrcp_count, width)
    #plt.barh()
    for i in range(len(ind)):
        plt.text(normal_count[i] + offset1, i, str(normal_count[i]))
        plt.text(hrcp_count[i] + offset2, i-width, str(hrcp_count[i]))

    plt.title(title)
    plt.ylabel(f'{col} value')
    plt.xlabel('Count')
    plt.yticks(ind-width/2, ytick_labels)
    plt.legend(['Normal', 'HRCP'])
    plt.savefig(basepath + 'plots/' + title + '.png')

def plot_all_combined_bars(basepath):
    datadf = pd.read_csv(basepath + 'fri_data_v2.csv')
    hrcp = datadf['HRCP'].to_numpy()
    cols = datadf.columns
    for col in cols[1:]:
        bar_plot_combined(hrcp, datadf[col].to_numpy(), f"Combined_{col}", basepath, col)
def plot_all(basepath):
    datadf = pd.read_csv(basepath + 'fri_data_v2.csv')
    fildf1 = datadf[datadf['HRCP'] == 0]
    fildf2 = datadf[datadf['HRCP'] == 1]
    cols = datadf.columns
    for col in cols[1:]:
        bar_plot(fildf1[col].to_numpy(), f"Normal_case_{col}", basepath, col)
        bar_plot(fildf2[col].to_numpy(), f"Abnormal_case_{col}", basepath, col)