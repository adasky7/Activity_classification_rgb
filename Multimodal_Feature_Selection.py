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
# from resultMetrics import plot_confusion_matrix
from sklearn import preprocessing, model_selection, svm, neighbors, ensemble
# from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from Features import features
import time
# import pickle
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel
from sklearn.pipeline import Pipeline




start_time = time.time()
"""
Load features from CSV file
"""
filename = 'CAD-601.csv'
data, act_label = features(filename)


"""
Removing features with low variance
"""
selector = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
selector.fit(data)
feats = selector.get_support(indices=True)
feats = [column for column in data[feats]]
FEATURES = pd.DataFrame(selector.transform(data))
FEATURES.columns = feats

"""
Normalize features using Standardization or z-score
"""
#FEATURES_norm = (FEATURES - FEATURES.mean()) / FEATURES.std()
#FEATURES_norm = FEATURES_norm.fillna(0)
FEATURES_norm = (FEATURES - FEATURES.min()) / (FEATURES.max() - FEATURES.min())
FEATURES_norm = FEATURES_norm.fillna(0)

"""
Feature selection using SelectKBest method
"""
feat_index_arry = np.zeros(len(FEATURES_norm.columns))
accuracy_array = np.zeros(len(FEATURES_norm.columns))
accuracy_array2 = np.zeros(len(FEATURES_norm.columns))

classifier = svm.LinearSVC()
classifier2 = neighbors.KNeighborsClassifier()
#plt.figure()
for m in range(1, len(FEATURES_norm.columns)+1):
    print('Count= %s' % m)
    selected_feat = SelectKBest(chi2, k=m).fit(FEATURES_norm, act_label)
    selected_feat1 = SelectKBest(chi2, k=m).fit_transform(FEATURES_norm, act_label)
    #print(selected_feat.get_support(indices=True))
    #print(selected_feat1)


    #train and tes data are split after feature selection
    train_act, test_act, train_label, test_label = model_selection.train_test_split(selected_feat1, act_label, test_size=0.2)
    print("Completed feature normalization")

    #SVM

    classifier.fit(train_act, train_label)
    _accuracy = classifier.score(test_act, test_label)
    print(_accuracy)
    feat_index_arry[m-1] = m
    accuracy_array[m-1] = _accuracy

    #KNN

    classifier2.fit(train_act, train_label)
    _accuracy2 = classifier2.score(test_act, test_label)
    print(_accuracy2)
    accuracy_array2[m - 1] = _accuracy2

# Plot
#svm
max_accuracy = np.amax(accuracy_array)
feat_ind_max_acc = feat_index_arry[np.argmax(accuracy_array)]
print('SVM Max accuracy = %s' % max_accuracy)
print('SVM Num of features at max accuracy = %s' % feat_ind_max_acc)

#knn
max_accuracy2 = np.amax(accuracy_array2)
feat_ind_max_acc2 = feat_index_arry[np.argmax(accuracy_array2)]
print('KNN Max accuracy = %s' % max_accuracy2)
print('KNN Num of features at max accuracy = %s' % feat_ind_max_acc2)


plt.xlim(0, 100)
plt.ylim(0, 1.2)
plt.title('Accuracy vs Features')

#svm plot
svm_plt = plt.plot(feat_index_arry, accuracy_array, 'r')
svm_max_point = plt.plot(feat_ind_max_acc, max_accuracy, 'bx')
plt.hlines(max_accuracy, 0, feat_ind_max_acc, linestyles='dashed', colors='b', linewidth=0.5)
plt.vlines(feat_ind_max_acc, 0, max_accuracy, linestyles='dashed', colors='b', linewidth=0.5)
#knn plot
knn_plt = plt.plot(feat_index_arry, accuracy_array2, 'k')
knn_max_point = plt.plot(feat_ind_max_acc2, max_accuracy2, 'gx')
plt.hlines(max_accuracy2, 0, feat_ind_max_acc2, linestyles='dashed', colors='g', linewidth=0.5)
plt.vlines(feat_ind_max_acc2, 0, max_accuracy2, linestyles='dashed', colors='g', linewidth=0.5)

plt.legend([svm_plt[0], knn_plt[0]], ['SVM', 'KNN'])
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.grid(linestyle='dotted', linewidth=0.3)
plt.show()


# knn_fig = fig.add_subplot(122)
# knn_fig.xlim(0, 100)
# knn_fig.ylim(0, 1.2)
# knn_fig.title('KNN - Accuracy vs Features')
# knn_fig.plot(feat_index_arry, accuracy_array2, 'r')
# knn_fig.hlines(max_accuracy2, 0, feat_ind_max_acc2, linestyles='dashed', colors='b', linewidth=0.5)
# knn_fig.vlines(feat_ind_max_acc2, 0, max_accuracy2, linestyles='dashed', colors='b', linewidth=0.5)
# #knn_fig.xlabel('Number of features')
# #knn_fig.ylabel('Accuracy')
# knn_fig.grid(linestyle='dotted', linewidth=0.3)



"""
split features for prediction test:
Actual training activity data = TRAIN_act
Training label = TRAIN_label
Test activity data = test_act
Test activity label = test_label

predicting new activity:
'pred_test' and 'pred_label' are used for predicting out the classifier
after training and testing are done
"""
#TRAIN_act, pred_test, TRAIN_label, pred_label = model_selection.train_test_split(train_act, train_label, test_size=0.1)
#discard rows in TRAIN_act with random activities
# training_data = pd.DataFrame(TRAIN_act)
# training_data['label'] = pd.DataFrame(TRAIN_label)
# training_data = training_data[training_data.label != 13]    #drop all rows with random activity data
# TRAIN_act = training_data.drop(['label'], axis=1)
# TRAIN_label = training_data.label




"""
SVM


classifier = svm.SVC(decision_function_shape='ovo', kernel='linear', probability=True)
classifier.fit(train_act, train_label)
_accuracy = classifier.score(test_act, test_label)
print(_accuracy)

joblib.dump(classifier, 'svm_model.pkl')    #save model to variable which will be used in classifying incoming kinect data


# predict outcome using prediction data
prediction = classifier.predict(test_act)
pred_prob = classifier.predict_proba(test_act)
print(pred_prob)
"""


# cnf_matrix = pd.crosstab(test_label, prediction, rownames=['Actual activity'], colnames=['Predicted activity'])
# #normalise confusion matrix
# cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)
# #multiply by 100 to express values in percentage
# cnf_matrix = cnf_matrix.multiply(100)
# print("Confusion matrix for %s activities: \n%s" % (filename, cnf_matrix))
# #Display report after classification (precision, recall and f-score)
# print("\nClassification report for %s activities: \n%s" % (filename, classification_report(test_label, prediction)))
#
# #convert column names to str for adjusting the ticks before plotting
# cnf_matrix.columns = cnf_matrix.columns.astype(str)
# cnf_matrix.index = cnf_matrix.index.astype(str)
# #print(cnf_matrix.index)
# for y in range(0, len(cnf_matrix.columns)):
#     cnf_matrix.columns.values[y] = 'A' + cnf_matrix.columns.values[y]
#     cnf_matrix.index.values[y] = 'A' + cnf_matrix.index.values[y]
# #plot confusion matrix
# plot_confusion_matrix(cnf_matrix)
# plt.show()

"""
K-nearest neighbour
"""
# classifier = neighbors.KNeighborsClassifier()
# classifier.fit(train_act, train_label)
# print("Training ended")
# _accuracy = classifier.score(test_act, test_label)
# print("KNN accuracy= %s" % _accuracy)
#
# #predict outcome using prediction data
# prediction = classifier.predict(test_act)
# #print(test_label)
# cnf_matrix = pd.crosstab(test_label, prediction, rownames=['Actual activity'], colnames=['Predicted activity'])
# #normalise confusion matrix
# cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)
# #multiply by 100 to express values in percentage
# cnf_matrix = cnf_matrix.multiply(100)
# print("Confusion matrix for %s activities: \n%s" % (filename, cnf_matrix))
# #Display report after classification (precision, recall and f-score)
# print("\nClassification report for %s activities: \n%s" % (filename, precision_recall_fscore_support(test_label, prediction)))
#
# #convert column names to str for adjusting the ticks before plotting
# cnf_matrix.columns = cnf_matrix.columns.astype(str)
# cnf_matrix.index = cnf_matrix.index.astype(str)
# #print(cnf_matrix.index)
# for y in range(0, len(cnf_matrix.columns)):
#     cnf_matrix.columns.values[y] = 'A' + cnf_matrix.columns.values[y]
#     cnf_matrix.index.values[y] = 'A' + cnf_matrix.index.values[y]
# #plot confusion matrix
# plot_confusion_matrix(cnf_matrix)
# plt.show()

"""
Random Forest
"""

# classifier = ensemble.RandomForestClassifier()
# classifier.fit(train_act, train_label)
# _accuracy = classifier.score(test_act, test_label)
# print(_accuracy)
#
# #predict outcome using prediction data
# prediction = classifier.predict(test_act)
# #print(test_label)
# cnf_matrix = pd.crosstab(test_label, prediction, rownames=['Actual activity'], colnames=['Predicted activity'])
# #normalise confusion matrix
# cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)
# #multiply by 100 to express values in percentage
# cnf_matrix = cnf_matrix.multiply(100)
# print("Confusion matrix for %s activities: \n%s" % (filename, cnf_matrix))
# #Display report after classification (precision, recall and f-score)
# print("\nClassification report for %s activities: \n%s" % (filename, precision_recall_fscore_support(test_label, prediction)))
#
# #convert column names to str for adjusting the ticks before plotting
# cnf_matrix.columns = cnf_matrix.columns.astype(str)
# cnf_matrix.index = cnf_matrix.index.astype(str)
# for y in range(0, len(cnf_matrix.columns)):
#     cnf_matrix.columns.values[y] = 'A' + cnf_matrix.columns.values[y]
#     cnf_matrix.index.values[y] = 'A' + cnf_matrix.index.values[y]
#
# #plot confusion matrix
# plot_confusion_matrix(cnf_matrix)
# plt.show()

print("... %s seconds..." % (time.time() - start_time))