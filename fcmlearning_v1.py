"""
This program is performs activity classification by applying a clustering algorithm (Fuzzy C-Means clustering)
to find similar patterns in an activity and group the activties for further prediction using an unseen data.

To run the program: a CSV file containing joint coordinates of activities and their corresponding labels is provided.
Data within the file = x,y,z coordinates of 15 joints, i.e. 3*15 = 45 columns. Plus a last column of activity label.

The number of clusters/classes/activities within the file is required
"""

from __future__ import division, print_function
import numpy as np
import pandas as pd
from pandas.tools.plotting import parallel_coordinates, scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.colors
import skfuzzy as fuzz
from sklearn import model_selection
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, f1_score


"""
Read activity data from csv file
"""
filename = 'CAD-601.csv'
df = pd.read_csv(filename)
act_label = pd.DataFrame(df.label)                                    #actual activity label

"""
Compute features for the activities by calculating euclidean distance between specified joint coordinates
"""
feat_1 = np.sqrt((df.rhandX - df.lhandX)**2 + (df.rhandY - df.lhandY)**2 + (df.rhandZ - df.lhandZ)**2)
feat_2 = np.sqrt((df.rhandX - df.headX)**2 + (df.rhandY - df.headY)**2 + (df.rhandZ - df.headZ)**2)
feat_3 = np.sqrt((df.lhandX - df.headX)**2 + (df.lhandY - df.headY)**2 + (df.lhandZ - df.headZ)**2)
feat_4 = np.sqrt((df.rhipX - df.rfootX)**2 + (df.rhipY - df.rfootY)**2 + (df.rhipZ - df.rfootZ)**2)
feat_5 = np.sqrt((df.lhipX - df.lfootX)**2 + (df.lhipY - df.lfootY)**2 + (df.lhipZ - df.lfootZ)**2)
feat_6 = np.sqrt((df.rshouldX - df.rfootX)**2 + (df.rshouldY - df.rfootY)**2 + (df.rshouldZ - df.rfootZ)**2)
feat_7 = np.sqrt((df.lshouldX - df.lfootX)**2 + (df.lshouldY - df.lfootY)**2 + (df.lshouldZ - df.lfootZ)**2)
feat_8 = np.sqrt((df.lhandX - df.lfootX)**2 + (df.lhandY - df.lfootY)**2 + (df.lhandZ - df.lfootZ)**2)
feat_9 = np.sqrt((df.rhandX - df.rfootX)**2 + (df.rhandY - df.rfootY)**2 + (df.rhandZ - df.rfootZ)**2)

"""
Create new data frame to hold the features
"""
FEATURES = pd.DataFrame({'feat1': feat_1, 'feat2': feat_2, 'feat3': feat_3, 'feat4': feat_4, 'feat5': feat_5, 'feat6': feat_6, 'feat7': feat_7, 'feat8': feat_8, 'feat9': feat_9})

"""
Normalize features to zero mean and unit variance
"""
FEATURES_norm = (FEATURES - FEATURES.mean()) / FEATURES.std()




"""
split data into training and testing data
"""
train_act, test_act, train_label, test_label = model_selection.train_test_split(FEATURES_norm, act_label, test_size=0.05)
'''
create new dataframe for data visualization and plot parallel coordinates
'''

plotData = test_act
#plotData['label'] = ('A' + test_label['label'].astype(str)).values
plotData['label'] = test_label.values
print('starting plot')
parallel_coordinates(plotData, 'label')
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'black', 'yellow', 'pink', 'indigo', 'khaki', 'gold', 'orange', 'grey', 'violet']
#palette = {1: 'red', 2: 'green', 3: 'blue', 4: 'cyan', 5: 'magenta', 6: 'black', 7: 'yellow', 8: 'white', 9: 'pink', 10: 'indigo', 11: 'khaki', 12: 'gold', 13: 'orange'}
#for c in np.nditer(plotData.label): colors.append(palette[int(c)])
plotData1 = test_act
plotData1['label'] = test_label.values
#scatter_matrix(plotData1, alpha=0.2, diagonal='kde', grid=True, marker='x')
plt.show()
print('end plot')


"""


train_act.to_csv('trainFeat.csv')
train_label.to_csv('trainLabel.csv')
test_act.to_csv('testFeat.csv')
test_label.to_csv('testLabel.csv')

'''
Begin Fuzzy c-means clustering using features computed
-declare the parameters to be used in clustering function
'''
#first create initial partitions to be used in the fuzzy function
act_list = train_label['label'].value_counts(sort=False).index.tolist() #get the total number of activities present in an activities input file
f_init_part = pd.DataFrame(np.zeros((len(train_label), len(act_list)), dtype=np.int64))   #create a dataframe of num of observations by number of labels to hold initial fuzzy partitions

#convert to numpy array for proper indexing
train_label = train_label.values
train_act = train_act.values
#loop to fill in values for the initial fuzzy partitions
for i in range(0, len(act_list)):
    for y in range(0, len(train_label)):
        if act_list[i] == train_label[y]:
            f_init_part.set_value(y, i, 1)

#initialise values of fuzzy clustering parameters
MaxIter = 1000
fuzz_coef = 1.4
error = 0.001
num_of_clusters = len(act_list)     #number of clusters required => number of classes of activities in the dataset

'''
substitute variables in fuzzy c-means function
'''
centers, f_partitions, ini_partitions, eucl_dist, obj_func, num_iter, fpc = fuzz._cluster.cmeans(train_act.transpose(), num_of_clusters, fuzz_coef, error=error, maxiter=MaxIter, init=(f_init_part.transpose()))
fuzzy_part = pd.DataFrame(f_partitions)         #copy final partition to dataframe
eucl_dist = pd.DataFrame(eucl_dist.transpose())

'''
Create an array 'trained labels' to hold activity labels after clustering.
number of observations = num of training samples
This is obtained by using the following 'for-loop'
'''
num_observ = len(train_label)                            #get the number of observations from the partitions result
trained_label = np.zeros(shape=(num_observ), dtype=np.int64)    #create an array which will hold the class number each observation belongs

'''
Assign class labels to observations after clustering by taking the maximum membership of each observation and assign the corresponding label
'''
for x in range(0, num_observ):
    for i in range(0, len(act_list)):
        if (fuzzy_part[x].argmax()) == i:
            cluster = act_list[i]
            #cluster = (fuzzy_part[x].argmax()) + 1  #this returns the class the max partition falls in by returning the row index
            trained_label[x] = cluster              #the class each observation falls in

#reshape the trained labels obtained after clustering and the original labels
trained_label = trained_label.reshape((len(trained_label), 1))
trained_label = np.concatenate((np.transpose(trained_label)))
train_label = (np.transpose(train_label)).reshape(-1)

'''
Obtain the confusion matrix which we use to obtain 'Purity'; obtained by taking the sum of (maximum values for each row of the confusion matrix) / (total number of values)
'''
cnf_matrix = pd.crosstab(trained_label, train_label, rownames=['cluster classified'], colnames=['Actual class of object'], margins=True)
print("Confusion matrix after fuzzy c_means: \n", cnf_matrix)

#purity of clustering


#normalized mutual information (NMI)
NMI = normalized_mutual_info_score(train_label, trained_label)
print("normalized mutual information = ", NMI)

#random index
RI = adjusted_rand_score(train_label, trained_label)
print("Random index = ", RI)

#adjusted mutual information
AMI = adjusted_mutual_info_score(train_label, trained_label)
print("Adjusted mutual information = ", AMI)

#f-measure score
#f_measure = f1_score(train_label, trained_label, average='sample')
#print("f-measure score = ", f_measure)

'''
Predict on test data
'''
#print(test_act.transpose().shape)
#test_part, test_ini_part, test_eucli, test_obj, test_iter, test_fpc = fuzz._cluster.cmeans_predict(test_act.transpose(),centers, fuzz_coef, error=error, maxiter=MaxIter)

"""