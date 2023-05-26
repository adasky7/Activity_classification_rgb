"""
This program extract features from Human activities skeleton joints data

"""


from __future__ import division, print_function
import numpy as np
import pandas as pd


def features(df):

    # """
    # Read activity data from csv file.
    # """
    # #filename = 'CAD-601.csv'
    # activity_df = pd.read_csv(filename)
    # df = activity_df.drop('label', 1)   #drop the column of activity label
    # act_label = activity_df.label      #to be uncommented if file contains a column of activity labels

    """
    offset data to have origin of points relative to the torso position
    uncomment this code if data is not offset before inputting into the function
    """
    torsoX = df.iloc[:, 6]
    torsoY = df.iloc[:, 7]
    torsoZ = df.iloc[:, 8]
    for x in range(0, (len(df.columns)), 3):
       df.iloc[:, x] = df.iloc[:, x] - torsoX
       df.iloc[:, (x+1)] = df.iloc[:, (x+1)] - torsoY
       df.iloc[:, (x + 2)] = df.iloc[:, (x + 2)] - torsoZ

    ##CAD-601.csv data is offset but torso coordinates remain unchanged. There we fill torso X,Y,Z with zeros
    #for i in range(6, 9):
    #    df.iloc[:, i] = np.zeros(len(df))
    ##print(df)

    """
    Spatial joint distance features:From activities data compute spatial joint distance features for the activities by calculating
    euclidean distance between specified joint coordinates
    """
    SJD_feat_1 = np.sqrt((df.rhandX - df.lhandX)**2 + (df.rhandY - df.lhandY)**2 + (df.rhandZ - df.lhandZ)**2)                  #distance between left and right hand
    SJD_feat_2 = np.sqrt((df.rhandX - df.headX)**2 + (df.rhandY - df.headY)**2 + (df.rhandZ - df.headZ)**2)                     #distance between right hand and head
    SJD_feat_3 = np.sqrt((df.lhandX - df.headX)**2 + (df.lhandY - df.headY)**2 + (df.lhandZ - df.headZ)**2)                     #distance between left hand and head
    SJD_feat_4 = np.sqrt((df.rhipX - df.rfootX)**2 + (df.rhipY - df.rfootY)**2 + (df.rhipZ - df.rfootZ)**2)                     #distance between right hip and right foot
    SJD_feat_5 = np.sqrt((df.lhipX - df.lfootX)**2 + (df.lhipY - df.lfootY)**2 + (df.lhipZ - df.lfootZ)**2)                     #distance between left hip and left foot
    SJD_feat_6 = np.sqrt((df.rshouldX - df.rfootX)**2 + (df.rshouldY - df.rfootY)**2 + (df.rshouldZ - df.rfootZ)**2)            #distance between right shoulder and right foot
    SJD_feat_7 = np.sqrt((df.lshouldX - df.lfootX)**2 + (df.lshouldY - df.lfootY)**2 + (df.lshouldZ - df.lfootZ)**2)            #distance between left shoulder and left foot
    SJD_feat_8 = np.sqrt((df.lhandX - df.lfootX)**2 + (df.lhandY - df.lfootY)**2 + (df.lhandZ - df.lfootZ)**2)                  #distance between left hand and left foot
    SJD_feat_9 = np.sqrt((df.rhandX - df.rfootX)**2 + (df.rhandY - df.rfootY)**2 + (df.rhandZ - df.rfootZ)**2)                  #distance between right hand and right foot
    #consider including euclidean distance of each joint to the torso center coordinates as a feature

    euclid_dist = pd.DataFrame({'SJD_feat_1': SJD_feat_1, 'SJD_feat_2': SJD_feat_2, 'SJD_feat_3': SJD_feat_3, 'SJD_feat_4': SJD_feat_4, 'SJD_feat_5': SJD_feat_5, 'SJD_feat_6': SJD_feat_6, 'SJD_feat_7': SJD_feat_7, 'SJD_feat_8': SJD_feat_8, 'SJD_feat_9': SJD_feat_9})


    """
    Temporal joint displacement : temporal location difference of same body joint in the current frame with respect to the prev frame
    """
    temp_joint_disp = df.diff()                             #calculate the difference between two frames of activity
    temp_joint_disp = temp_joint_disp.fillna(0)         #replace NaN values with 0 (i.e. the first row whose difference initially = 0)
    temp_joint_disp = temp_joint_disp.rename(columns={"headX": "temp_headX", "headY": "temp_headY", "headZ": "temp_headZ", "neckX": "temp_neckX", "neckY": "temp_neckY", "neckZ": "temp_neckZ",
                                                      "torsoX": "temp_torsoX", "torsoY": "temp_torsoY", "torsoZ": "temp_torsoZ", "lshouldX": "temp_lshouldX", "lshouldY": "temp_lshouldY", "lshouldZ": "temp_lshouldZ",
                                                      "lelbowX": "temp_lelbowX", "lelbowY": "temp_lelbowY", "lelbowZ": "temp_lelbowZ", "rshouldX": "temp_rshouldX", "rshouldY": "temp_rshouldY", "rshouldZ": "temp_rshouldZ",
                                                      "relbowX": "temp_relbowX", "relbowY": "temp_relbowY", "relbowZ": "temp_relbowZ", "lhipX": "temp_lhipX", "lhipY": "temp_lhipY", "lhipZ": "temp_lhipZ", "lkneeX": "temp_lkneeX",
                                                      "lkneeY": "temp_lkneeY", "lkneeZ": "temp_lkneeZ", "rhipX": "temp_rhipX", "rhipY": "temp_rhipY", "rhipZ": "temp_rhipZ", "rkneeX": "temp_rkneeX", "rkneeY": "temp_rkneeY",
                                                      "rkneeZ": "temp_rkneeZ", "lhandX": "temp_lhandX", "lhandY": "temp_lhandY", "lhandZ": "temp_lhandZ", "rhandX": "temp_rhandX", "rhandY": "temp_rhandY", "rhandZ": "temp_rhandZ",
                                                      "lfootX": "temp_lfootX", "lfootY": "temp_lfootY", "lfootZ": "temp_lfootZ", "rfootX": "temp_rfootX", "rfootY": "temp_rfootY", "rfootZ": "temp_rfootZ"})

    """
    Long term temporal joint displacement: temporal location difference of joints between the current frame (frame n) and the initial frame (frame 1)
    """
    long_temp_disp = df - df.iloc[0]
    long_temp_disp = long_temp_disp.rename(columns={"headX": "Ltemp_headX", "headY": "Ltemp_headY", "headZ": "Ltemp_headZ", "neckX": "Ltemp_neckX", "neckY": "Ltemp_neckY", "neckZ": "Ltemp_neckZ",
                                                    "torsoX": "Ltemp_torsoX", "torsoY": "Ltemp_torsoY", "torsoZ": "Ltemp_torsoZ", "lshouldX": "Ltemp_lshouldX", "lshouldY": "Ltemp_lshouldY", "lshouldZ": "Ltemp_lshouldZ",
                                                    "lelbowX": "Ltemp_lelbowX", "lelbowY": "Ltemp_lelbowY", "lelbowZ": "Ltemp_lelbowZ", "rshouldX": "Ltemp_rshouldX", "rshouldY": "Ltemp_rshouldY", "rshouldZ": "Ltemp_rshouldZ",
                                                    "relbowX": "Ltemp_relbowX", "relbowY": "Ltemp_relbowY", "relbowZ": "Ltemp_relbowZ", "lhipX": "Ltemp_lhipX", "lhipY": "Ltemp_lhipY", "lhipZ": "Ltemp_lhipZ", "lkneeX": "Ltemp_lkneeX",
                                                    "lkneeY": "Ltemp_lkneeY", "lkneeZ": "Ltemp_lkneeZ", "rhipX": "Ltemp_rhipX", "rhipY": "Ltemp_rhipY", "rhipZ": "Ltemp_rhipZ", "rkneeX": "Ltemp_rkneeX", "rkneeY": "Ltemp_rkneeY",
                                                    "rkneeZ": "Ltemp_rkneeZ", "lhandX": "Ltemp_lhandX", "lhandY": "Ltemp_lhandY", "lhandZ": "Ltemp_lhandZ", "rhandX": "Ltemp_rhandX", "rhandY": "Ltemp_rhandY", "rhandZ": "Ltemp_rhandZ",
                                                    "lfootX": "Ltemp_lfootX", "lfootY": "Ltemp_lfootY", "lfootZ": "Ltemp_lfootZ", "rfootX": "Ltemp_rfootX", "rfootY": "Ltemp_rfootY", "rfootZ": "Ltemp_rfootZ"})


    """
    More features
    """
    ###

    """
    Combine all features into one feature dataframe
    """
    comb_feat = pd.concat([euclid_dist, temp_joint_disp, long_temp_disp], axis=1)

    """
    Normalize features to zero mean and unit variance
    """
    comb_features = (comb_feat - comb_feat.mean()) / comb_feat.std()
    comb_features = comb_features.fillna(0)

    return comb_features#, act_label