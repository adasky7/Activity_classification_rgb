"""
This program extract features from Human activities skeleton joints data

"""


from __future__ import division, print_function
import numpy as np
import pandas as pd

"""
Feature extraction from each activity
"""

def features(dataframe):
    """
    Select columns for each activity
    """
    """
    for i in range(0, len(dataframe.columns), 3):
        if i == (len(dataframe.columns) - 1):
            break
        dataframe.iloc[:, i] = dataframe.iloc[:, i] - dataframe.iloc[:, 6]
        dataframe.iloc[:, (i+1)] = dataframe.iloc[:, (i+1)] - dataframe.iloc[:, 7]
        dataframe.iloc[:, (i+2)] = dataframe.iloc[:, (i+2)] - dataframe.iloc[:, 8]
    print(dataframe)
    """
    for i in range(0, len(dataframe.columns)):
        if i == (len(dataframe.columns) - 1):
            break
        dataframe.iloc[:, i] = dataframe.iloc[:, i] * 10

    dataframe = dataframe.reset_index(drop=True) # reset row index when getting data from a file containing multiple activities
    df = dataframe.drop('label', axis=1)   # drop the column of activity label
    act_label = dataframe.label      # to be uncommented if file contains a column of activity labels


    """
    Spatial joint distance features:From activities data compute spatial joint distance features for the activities by calculating
    euclidean distance between specified joint coordinates
    """
    SJD_feat_1 = np.sqrt((df.rhandX - df.lhandX)**2 + (df.rhandY - df.lhandY)**2 + (df.rhandZ - df.lhandZ)**2)                  # distance between left and right hand
    SJD_feat_2 = np.sqrt((df.rhandX - df.headX)**2 + (df.rhandY - df.headY)**2 + (df.rhandZ - df.headZ)**2)                     # distance between right hand and head
    SJD_feat_3 = np.sqrt((df.lhandX - df.headX)**2 + (df.lhandY - df.headY)**2 + (df.lhandZ - df.headZ)**2)                     # distance between left hand and head
    SJD_feat_4 = np.sqrt((df.rhipX - df.rfootX)**2 + (df.rhipY - df.rfootY)**2 + (df.rhipZ - df.rfootZ)**2)                     # distance between right hip and right foot
    SJD_feat_5 = np.sqrt((df.lhipX - df.lfootX)**2 + (df.lhipY - df.lfootY)**2 + (df.lhipZ - df.lfootZ)**2)                     # distance between left hip and left foot
    SJD_feat_6 = np.sqrt((df.rshouldX - df.rfootX)**2 + (df.rshouldY - df.rfootY)**2 + (df.rshouldZ - df.rfootZ)**2)            # distance between right shoulder and right foot
    SJD_feat_7 = np.sqrt((df.lshouldX - df.lfootX)**2 + (df.lshouldY - df.lfootY)**2 + (df.lshouldZ - df.lfootZ)**2)            # distance between left shoulder and left foot
    SJD_feat_8 = np.sqrt((df.lhandX - df.lfootX)**2 + (df.lhandY - df.lfootY)**2 + (df.lhandZ - df.lfootZ)**2)                  # distance between left hand and left foot
    SJD_feat_9 = np.sqrt((df.rhandX - df.rfootX)**2 + (df.rhandY - df.rfootY)**2 + (df.rhandZ - df.rfootZ)**2)                  # distance between right hand and right foot
    # consider including euclidean distance of each joint to the torso center coordinates as a feature

    euclid_dist = pd.DataFrame({'SJD_feat_1': SJD_feat_1, 'SJD_feat_2': SJD_feat_2, 'SJD_feat_3': SJD_feat_3, 'SJD_feat_4': SJD_feat_4, 'SJD_feat_5': SJD_feat_5, 'SJD_feat_6': SJD_feat_6, 'SJD_feat_7': SJD_feat_7, 'SJD_feat_8': SJD_feat_8, 'SJD_feat_9': SJD_feat_9})


    """
    Temporal joint displacement : temporal location difference of same body joint in the current frame with respect to the prev frame
    """
    temp_joint_disp = df.diff()                             # calculate the difference between two frames of activity
    temp_joint_disp = temp_joint_disp.fillna(0)         # replace NaN values with 0 (i.e. the first row whose difference initially = 0)
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
    Mean across each frame of activity
    """
    Mean = df.mean(axis=0)
    Mean = pd.DataFrame(Mean)
    Mean = Mean.T
    Mean = Mean.rename(columns={"headX": "mean_headX", "headY": "mean_headY", "headZ": "mean_headZ", "neckX": "mean_neckX", "neckY": "mean_neckY", "neckZ": "mean_neckZ",
                                                    "torsoX": "mean_torsoX", "torsoY": "mean_torsoY", "torsoZ": "mean_torsoZ", "lshouldX": "mean_lshouldX", "lshouldY": "mean_lshouldY", "lshouldZ": "mean_lshouldZ",
                                                    "lelbowX": "mean_lelbowX", "lelbowY": "mean_lelbowY", "lelbowZ": "mean_lelbowZ", "rshouldX": "mean_rshouldX", "rshouldY": "mean_rshouldY", "rshouldZ": "mean_rshouldZ",
                                                    "relbowX": "mean_relbowX", "relbowY": "mean_relbowY", "relbowZ": "mean_relbowZ", "lhipX": "mean_lhipX", "lhipY": "mean_lhipY", "lhipZ": "mean_lhipZ", "lkneeX": "mean_lkneeX",
                                                    "lkneeY": "mean_lkneeY", "lkneeZ": "mean_lkneeZ", "rhipX": "mean_rhipX", "rhipY": "mean_rhipY", "rhipZ": "mean_rhipZ", "rkneeX": "mean_rkneeX", "rkneeY": "mean_rkneeY",
                                                    "rkneeZ": "mean_rkneeZ", "lhandX": "mean_lhandX", "lhandY": "mean_lhandY", "lhandZ": "mean_lhandZ", "rhandX": "mean_rhandX", "rhandY": "mean_rhandY", "rhandZ": "mean_rhandZ",
                                                    "lfootX": "mean_lfootX", "lfootY": "mean_lfootY", "lfootZ": "mean_lfootZ", "rfootX": "mean_rfootX", "rfootY": "mean_rfootY", "rfootZ": "mean_rfootZ"})
    meanFeat = pd.DataFrame(df.values - Mean.values,
                            columns=Mean.columns)  # combine mean and df in order to find the difference of each column from its mean

    """
    Variance across each sample
    """
    Variance = df.var(axis=0)
    Variance = pd.DataFrame(Variance)
    Variance = Variance.T
    Variance = Variance.rename(columns={"headX": "var_headX", "headY": "var_headY", "headZ": "var_headZ", "neckX": "var_neckX", "neckY": "var_neckY", "neckZ": "var_neckZ",
                                                    "torsoX": "var_torsoX", "torsoY": "var_torsoY", "torsoZ": "var_torsoZ", "lshouldX": "var_lshouldX", "lshouldY": "var_lshouldY", "lshouldZ": "var_lshouldZ",
                                                    "lelbowX": "var_lelbowX", "lelbowY": "var_lelbowY", "lelbowZ": "var_lelbowZ", "rshouldX": "var_rshouldX", "rshouldY": "var_rshouldY", "rshouldZ": "var_rshouldZ",
                                                    "relbowX": "var_relbowX", "relbowY": "var_relbowY", "relbowZ": "var_relbowZ", "lhipX": "var_lhipX", "lhipY": "var_lhipY", "lhipZ": "var_lhipZ", "lkneeX": "var_lkneeX",
                                                    "lkneeY": "var_lkneeY", "lkneeZ": "var_lkneeZ", "rhipX": "var_rhipX", "rhipY": "var_rhipY", "rhipZ": "var_rhipZ", "rkneeX": "var_rkneeX", "rkneeY": "var_rkneeY",
                                                    "rkneeZ": "var_rkneeZ", "lhandX": "var_lhandX", "lhandY": "var_lhandY", "lhandZ": "var_lhandZ", "rhandX": "var_rhandX", "rhandY": "var_rhandY", "rhandZ": "var_rhandZ",
                                                    "lfootX": "var_lfootX", "lfootY": "var_lfootY", "lfootZ": "var_lfootZ", "rfootX": "var_rfootX", "rfootY": "var_rfootY", "rfootZ": "var_rfootZ"})
    varFeat = pd.DataFrame(df.values - Variance.values, columns=Variance.columns)

    """
    Standard deviation across each sample
    """
    Std_deviatn = df.std(axis=0)
    Std_deviatn = pd.DataFrame(Std_deviatn)
    Std_deviatn = Std_deviatn.T
    Std_deviatn = Std_deviatn.rename(columns={"headX": "std_headX", "headY": "std_headY", "headZ": "std_headZ", "neckX": "std_neckX", "neckY": "std_neckY", "neckZ": "std_neckZ",
                                                    "torsoX": "std_torsoX", "torsoY": "std_torsoY", "torsoZ": "std_torsoZ", "lshouldX": "std_lshouldX", "lshouldY": "std_lshouldY", "lshouldZ": "std_lshouldZ",
                                                    "lelbowX": "std_lelbowX", "lelbowY": "std_lelbowY", "lelbowZ": "std_lelbowZ", "rshouldX": "std_rshouldX", "rshouldY": "std_rshouldY", "rshouldZ": "std_rshouldZ",
                                                    "relbowX": "std_relbowX", "relbowY": "std_relbowY", "relbowZ": "std_relbowZ", "lhipX": "std_lhipX", "lhipY": "std_lhipY", "lhipZ": "std_lhipZ", "lkneeX": "std_lkneeX",
                                                    "lkneeY": "std_lkneeY", "lkneeZ": "std_lkneeZ", "rhipX": "std_rhipX", "rhipY": "std_rhipY", "rhipZ": "std_rhipZ", "rkneeX": "std_rkneeX", "rkneeY": "std_rkneeY",
                                                    "rkneeZ": "std_rkneeZ", "lhandX": "std_lhandX", "lhandY": "std_lhandY", "lhandZ": "std_lhandZ", "rhandX": "std_rhandX", "rhandY": "std_rhandY", "rhandZ": "std_rhandZ",
                                                    "lfootX": "std_lfootX", "lfootY": "std_lfootY", "lfootZ": "std_lfootZ", "rfootX": "std_rfootX", "rfootY": "std_rfootY", "rfootZ": "std_rfootZ"})
    stdFeat = pd.DataFrame(df.values - Std_deviatn.values, columns=Std_deviatn.columns)

    """
    Skewness of each frame
    """
    Skewness = df.skew(axis=0)
    Skewness = pd.DataFrame(Skewness)
    Skewness = Skewness.T
    Skewness = Skewness.rename(columns={"headX": "skw_headX", "headY": "skw_headY", "headZ": "skw_headZ", "neckX": "skw_neckX", "neckY": "skw_neckY", "neckZ": "skw_neckZ",
                                                    "torsoX": "skw_torsoX", "torsoY": "skw_torsoY", "torsoZ": "skw_torsoZ", "lshouldX": "skw_lshouldX", "lshouldY": "skw_lshouldY", "lshouldZ": "skw_lshouldZ",
                                                    "lelbowX": "skw_lelbowX", "lelbowY": "skw_lelbowY", "lelbowZ": "skw_lelbowZ", "rshouldX": "skw_rshouldX", "rshouldY": "skw_rshouldY", "rshouldZ": "skw_rshouldZ",
                                                    "relbowX": "skw_relbowX", "relbowY": "skw_relbowY", "relbowZ": "skw_relbowZ", "lhipX": "skw_lhipX", "lhipY": "skw_lhipY", "lhipZ": "skw_lhipZ", "lkneeX": "skw_lkneeX",
                                                    "lkneeY": "skw_lkneeY", "lkneeZ": "skw_lkneeZ", "rhipX": "skw_rhipX", "rhipY": "skw_rhipY", "rhipZ": "skw_rhipZ", "rkneeX": "skw_rkneeX", "rkneeY": "skw_rkneeY",
                                                    "rkneeZ": "skw_rkneeZ", "lhandX": "skw_lhandX", "lhandY": "skw_lhandY", "lhandZ": "skw_lhandZ", "rhandX": "skw_rhandX", "rhandY": "skw_rhandY", "rhandZ": "skw_rhandZ",
                                                    "lfootX": "skw_lfootX", "lfootY": "skw_lfootY", "lfootZ": "skw_lfootZ", "rfootX": "skw_rfootX", "rfootY": "skw_rfootY", "rfootZ": "skw_rfootZ"})
    skwFeat = pd.DataFrame(df.values - Skewness.values, columns=Skewness.columns)

    """
    Kurtosis
    """
    Kurtosis = df.kurtosis(axis=0)
    Kurtosis = pd.DataFrame(Kurtosis)
    Kurtosis = Kurtosis.T
    Kurtosis = Kurtosis.rename(columns={"headX": "kur_headX", "headY": "kur_headY", "headZ": "kur_headZ", "neckX": "kur_neckX", "neckY": "kur_neckY", "neckZ": "kur_neckZ",
                                                    "torsoX": "kur_torsoX", "torsoY": "kur_torsoY", "torsoZ": "kur_torsoZ", "lshouldX": "kur_lshouldX", "lshouldY": "kur_lshouldY", "lshouldZ": "kur_lshouldZ",
                                                    "lelbowX": "kur_lelbowX", "lelbowY": "kur_lelbowY", "lelbowZ": "kur_lelbowZ", "rshouldX": "kur_rshouldX", "rshouldY": "kur_rshouldY", "rshouldZ": "kur_rshouldZ",
                                                    "relbowX": "kur_relbowX", "relbowY": "kur_relbowY", "relbowZ": "kur_relbowZ", "lhipX": "kur_lhipX", "lhipY": "kur_lhipY", "lhipZ": "kur_lhipZ", "lkneeX": "kur_lkneeX",
                                                    "lkneeY": "kur_lkneeY", "lkneeZ": "kur_lkneeZ", "rhipX": "kur_rhipX", "rhipY": "kur_rhipY", "rhipZ": "kur_rhipZ", "rkneeX": "kur_rkneeX", "rkneeY": "kur_rkneeY",
                                                    "rkneeZ": "kur_rkneeZ", "lhandX": "kur_lhandX", "lhandY": "kur_lhandY", "lhandZ": "kur_lhandZ", "rhandX": "kur_rhandX", "rhandY": "kur_rhandY", "rhandZ": "kur_rhandZ",
                                                    "lfootX": "kur_lfootX", "lfootY": "kur_lfootY", "lfootZ": "kur_lfootZ", "rfootX": "kur_rfootX", "rfootY": "kur_rfootY", "rfootZ": "kur_rfootZ"})
    kurFeat = pd.DataFrame(df.values - Kurtosis.values, columns=Kurtosis.columns)

    """
    More Features...
    """

    """
    Combine all features into one feature dataframe
    """
    comb_feat = pd.concat([euclid_dist, temp_joint_disp, long_temp_disp, meanFeat, varFeat, stdFeat, skwFeat, kurFeat, act_label], axis=1)


    return comb_feat


"""
Compute features for all persons and all activities (excluding the random activity)
"""
def compute_features(filename): # pass an array holding names of all files containing activity data
    num_of_files = len(filename)    #hold number of files containing activities data

    # read file into a dataframe
    for i in range(0, num_of_files):
        activity_df = pd.read_csv(filename[i])

        # for loop to handle all activities from CAD-60 dataset
        # The range stands for the activity labels or classes 1-12 plus 13 for random activity
        num_of_classes = activity_df.label.max()

        #CAD-60 data segmentation (delete for own data)
        bathroom = [1, 2, 3, 13]
        bedroom = [4, 5, 6, 13]
        kitchen = [5, 7, 8, 6, 13]
        living_room = [4, 5, 9, 10, 13]
        office = [4, 11, 5, 12, 13]
        mydata = [1, 2, 3, 4]

        current_scene = mydata
        #for x in range(1, num_of_classes):
        for x in range(0, len(current_scene)):
            dataframe = activity_df[activity_df.label == current_scene[x]]
            Ex_feat = features(dataframe)
            if (x == 0) and (i == 0):
                feat = Ex_feat
            else:
                feat = pd.concat([feat, Ex_feat], axis=0)

    feat = feat.reset_index(drop=True)
    print('Completed feature extraction for %s activities' % current_scene)

    return feat
# file = ['/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person1.csv']
# compute_features(file)

"""
Activity labels for CAD-60:
Rinsing Mouth   -   1
Brusthing Teeth -   2
Wearing Lenses  -   3
Talking on phone-   4
Drinking water  -   5
Open pill cont  -   6
Cooking (chop)  -   7
Cooking (stir)  -   8
Talking on couch-   9
Relax on couch  -   10
Write on board  -   11
Work on comp    -   12
Random+ still   -   13
"""