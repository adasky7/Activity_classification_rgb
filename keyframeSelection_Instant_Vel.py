"""
This program extracts frames (aka poses) from human skeleton data from which
kinetic energy of each frame is computed and plotted. We obtain key poses from the lowest
points or thresholds of kinetic energy using the formula K.E = (1/2)(m*(v)^2)

"""


from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, sys
from scipy.signal import savgol_filter
from scipy.fftpack import fft, rfft, irfft
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema

# set program time
start_time = time.time()
start_time_local = time.time()

# file name
file = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person3still.csv'

# Read data from .csv file containing activity data
activity_df = pd.read_csv(file)

# number of classes = number of activities within each file
num_of_classes = activity_df.label.max()

# for each class, extract joint coordinates for processing
# to calculate K.E 1/2(v^2)
for x in range(1, (num_of_classes + 1)):
    class_df = activity_df[activity_df.label == x]
    """
    calculating velocity of each joint dimension dist/time
    using a sampling time interval of 0.033 secs
    """
    class_label = class_df.label
    class_df = class_df.drop('label', axis=1)   # drop column with class label

    #class_df = class_df.rolling(window=30).mean()         # simple moving average to smoothing data
    coord_dist = class_df.diff().fillna(0)                # diff between each frame and its preceding frame and fill NaNs with 0
    coord_dist = coord_dist.rolling(window=30).mean()
    # compute joint distances over time
    coord_dist = coord_dist ** 2                          # square each value used to compute euclidean distance of a joint movement over time
    # print("coordinate distance %s" % coord_dist)
    for i in range(0, len(coord_dist.columns), 3):
        euclid_dist = np.sqrt(coord_dist.iloc[:, i] + coord_dist.iloc[:, (i+1)] + coord_dist.iloc[:, (i+2)])    # from second frame calculate each joint distance across each frame
        if i == 0:
            joint_dist = euclid_dist
        else:
            joint_dist = pd.concat([joint_dist, euclid_dist], axis=1)
    velocity2 = ((joint_dist / 100)) #/ 0.033)      # convert to meter / second
    K_E = (((velocity2.sum(axis=1) ** 2) / 2).reset_index(drop=True))
    K_E_sma = K_E.rolling(window=15).mean()
    K_E_ema = K_E.rolling(window=30).mean()

    # plot kinetic energy vs frame number
    cumu_diff_KE = K_E - K_E.loc[30]
    diff_sum = cumu_diff_KE.loc[30:len(cumu_diff_KE)].sum()

    if x == 5:
        plt.figure(x)
        idx = np.argwhere(np.diff(np.sign(K_E_sma - K_E_ema)) != 0).reshape(-1) + 0 #find the crossover points of the moving averages
        print(idx)
        #raw_plt = plt.plot(K_E.index, K_E, color='black')
        sh_sma = plt.plot(K_E.index[30:len(K_E.index)], K_E_sma[30:len(K_E.index)], color='red')
        long_sma = plt.plot(K_E.index[30:len(K_E.index)], K_E_ema[30:len(K_E.index)], color='green')
        plt.plot(K_E.index[idx], K_E_ema[idx], 'bo')
        plt.legend([sh_sma[0], long_sma[0]], ['Short Moving Av. = 15', 'Long Moving Av. = 30'])
        plt.xlabel('frame number')
        plt.ylabel('Kinetic energy')
        plt.show()
        sys.exit()
plt.show()
sys.exit()





























