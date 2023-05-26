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

# set program time
start_time = time.time()
start_time_local = time.time()

# file name
file = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person1still.csv'

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
    #class_df = (class_df - class_df.min()) / (class_df.max() - class_df.min())  # scale the data
    coord_dist = class_df.rolling(window=30).mean()#(class_df - class_df.iloc[0]).fillna(0)                # diff from each frame to the initial frame
    # compute joint distances over time
    coord_dist = coord_dist ** 2    # square each value used to compute euclidean distance of a joint movement over time
    # print("coordinate distance %s" % coord_dist)
    for i in range(0, len(coord_dist.columns), 3):
        euclid_dist = np.sqrt(coord_dist.iloc[:, i] + coord_dist.iloc[:, (i+1)] + coord_dist.iloc[:, (i+2)])    # from second frame calculate euclidean distance from one position to another of each joint movement
        if i == 0:
            joint_dist = euclid_dist
        else:
            joint_dist = pd.concat([joint_dist, euclid_dist], axis=1)
    joint_dist = joint_dist / 1000      # calculate cumulative sum of distance from initial frame to current frame and convert distance from millimeters to meters
    joint_dist = joint_dist.reset_index(drop=True)
    joint_dist = joint_dist - joint_dist.loc[29]
    #print(joint_dist)
    # compute velocity of movement for each joint in all samples (for each sample, divide joint distance by change in time)
    #for y in range(0, len(joint_dist.index)):
    #    joint_dist.iloc[y] = joint_dist.iloc[y] / (0.033 * y)
    velocity = joint_dist.fillna(0)
    velocity2 = velocity ** 2      # square the velocity
    K_E = ((velocity.sum(axis=1) ** 2) / 2).reset_index(drop=True)

    # print("K.E for pose %s" % K_E)
    # plot kinetic energy vs frame number
    #print(velocity)
    if x == 5:
        plt.figure(x)
        plt.plot(K_E.index, velocity.sum(axis=1), '-')
        plt.plot(K_E.index, K_E, 'r')
        plt.xlabel('frame number')
        plt.ylabel('Kinetic energy')
        plt.show()
        sys.exit()




























