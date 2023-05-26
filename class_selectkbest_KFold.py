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
from sklearn import preprocessing, model_selection, svm, neighbors
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from Features_extended import compute_features
import time, sys
# import pickle
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel
from sklearn.pipeline import Pipeline, make_pipeline
from threading import Thread, _start_new_thread
import traceback
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier


#set program time
start_time = time.time()
start_time_local = time.time()

# control system process time
def pretty_print(message, progress, reset = False):

    global start_time_local

    if (reset):
        start_time_local = time.time()

    current_time = time.time() - start_time
    current_time_local = time.time() - start_time_local
    print("G[%.3f] L[%.3f] [%.0f%%]: %s" % (current_time, current_time_local, progress, message))


"""
Import features computed
"""
file1 = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person1.csv'
file2 = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person2.csv'
file3 = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person3.csv'
file4 = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person4.csv'
CAD_60 = [file1, file2, file3, file4]

ex_features = compute_features(CAD_60)
# ex_features.to_csv('myresults.csv', sep=',')
data = ex_features.drop('label', 1)   # drop the column of activity label
data = data.fillna(method='ffill')
# print(data.isnull().any())
act_label = ex_features.label
print('Completed feature extraction')

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
# Standardization normalization (zero mean and unit variance)
# FEATURES_norm = (FEATURES - FEATURES.mean()) / FEATURES.std()
# FEATURES_norm = FEATURES_norm.fillna(0)

# Min-Max normalization
FEATURES_norm = (FEATURES - FEATURES.min()) / (FEATURES.max() - FEATURES.min())
FEATURES_norm = FEATURES_norm.fillna(0)
print("Completed feature normalization")

# declaration of cross validation method used (K-Fold)
kf = model_selection.KFold(n_splits=5)
kfold_count = 1             #keep count of kfold validation iteration
print("number of cross validation splits initialized = %s ....." % kf.get_n_splits(FEATURES_norm))

"""
Variable declaration used in feature selection and classification
"""
feat_index_arry = np.zeros(len(FEATURES_norm.columns))
accuracy_array = np.zeros(len(FEATURES_norm.columns))
accuracy_array2 = np.zeros(len(FEATURES_norm.columns))
accuracy_array3 = np.zeros(len(FEATURES_norm.columns))
accuracy_array4 = np.zeros(len(FEATURES_norm.columns))

classifier = svm.LinearSVC()
classifier2 = neighbors.KNeighborsClassifier(n_jobs=-1)
classifier3 = RandomForestClassifier(n_jobs=-1)

lengthOfFeatures = len(FEATURES_norm.columns) + 1


"""
---------------------Classification functions-----------------------------------------------------
"""


#setup svm classifier
def svm_class(value, selected_feat, trainA, trainL, testA, testL):
    global accuracy_array

    pretty_print('Classifier: SVM pipeline with selected feature', progress, True)
    # SVM pipeline with selected feature
    svm_pipe = make_pipeline(selected_feat, classifier)
    pretty_print('Classifier: A', progress, False)
    svm_pipe.fit(trainA, trainL)
    pretty_print('Classifier: B', progress, False)
    _accuracy = svm_pipe.score(testA, testL)
    # print('SVM %s' % _accuracy)
    pretty_print('SVM %s' % _accuracy, progress)
    accuracy_array[value - 1] = _accuracy

    # print feature indices for each round
    # pretty_print('Feature indices for each round', progress)
    # index = svm_pipe.named_steps['selectkbest'].get_support(indices=True)
    # print(index)


#setup knn classifier
def knn_class(value, selected_feat, trainA, trainL, testA, testL):
    global accuracy_array2

    pretty_print('Classifier: KNN pipeline with selected feature', progress, True)
    # KNN pipeline with selected feature
    knn_pipe = make_pipeline(selected_feat, classifier2)
    knn_pipe.fit(trainA, trainL)
    _accuracy2 = knn_pipe.score(testA, testL)
    # print('KNN %s' % _accuracy2)
    pretty_print('KNN %s' % _accuracy2, progress)
    accuracy_array2[value - 1] = _accuracy2


# setup random forest classifier
def rand_forest_class(value, selected_feat, trainA, trainL, testA, testL):
    global accuracy_array3
    try:
        pretty_print('Classifier: Random forest pipeline with selected feature', progress, True)
        # Random forest pipeline with selected feature
        random_f_pipe = make_pipeline(selected_feat, classifier3)
        random_f_pipe.fit(trainA, trainL)
        _accuracy3 = random_f_pipe.score(testA, testL)
        # print('Random Forest %s' % _accuracy3)
        pretty_print('Random Forest %s' % _accuracy3, progress)
        accuracy_array3[value - 1] = _accuracy3
    except:
        print("------------------")
        print(selected_feat)
        print("------------------")
        traceback.print_exception()
        dontdoabsolutelyanythingelseplease=1

def sgd_class(value, selected_feat, trainA, trainL, testA, testL):
    #pretty_print('SGD classifier', progress, True)
    print('SGD classifier')
    rbf_feature = RBFSampler(gamma=1, random_state=1)
    train_features = rbf_feature.fit_transform(trainA)
    clf = SGDClassifier()
    clf.fit(train_features, trainL)
    test_features = rbf_feature.fit_transform(testA)
    _accuracy4 = clf.score(test_features, testL)
    print('SGD kernelized feature accuracy %s' % _accuracy4)
    pretty_print('SGD kernelized feature accuracy %s' % _accuracy4, progress)
    accuracy_array4[value - 1] = _accuracy4

"""
-----------------------------Plot functions-------------------------------------------------------
"""


def svm_plot():
    global accuracy_array
    global feat_index_arry

    # svm accuracy values
    max_accuracy = np.amax(accuracy_array)
    feat_ind_max_acc = feat_index_arry[np.argmax(accuracy_array)]
    print('SVM Max accuracy = %s' % max_accuracy)
    print('SVM Num of features at max accuracy = %s' % feat_ind_max_acc)

    # svm plot
    svm_plt = plt.plot(feat_index_arry, accuracy_array, 'r')
    svm_max_point = plt.plot(feat_ind_max_acc, max_accuracy, 'bx')
    plt.hlines(max_accuracy, 0, feat_ind_max_acc, linestyles='dashed', colors='b', linewidth=0.5)
    plt.vlines(feat_ind_max_acc, 0, max_accuracy, linestyles='dashed', colors='b', linewidth=0.5)

    return svm_plt


def knn_plot():
    global accuracy_array2
    global feat_index_arry

    # knn accuracy values
    max_accuracy2 = np.amax(accuracy_array2)
    feat_ind_max_acc2 = feat_index_arry[np.argmax(accuracy_array2)]
    print('KNN Max accuracy = %s' % max_accuracy2)
    print('KNN Num of features at max accuracy = %s' % feat_ind_max_acc2)

    # knn plot
    knn_plt = plt.plot(feat_index_arry, accuracy_array2, 'k')
    knn_max_point = plt.plot(feat_ind_max_acc2, max_accuracy2, 'gx')
    plt.hlines(max_accuracy2, 0, feat_ind_max_acc2, linestyles='dashed', colors='g', linewidth=0.5)
    plt.vlines(feat_ind_max_acc2, 0, max_accuracy2, linestyles='dashed', colors='g', linewidth=0.5)

    return knn_plt


def randomF_plot():
    global accuracy_array3
    global feat_index_arry

    # Random forest accuracy values
    max_accuracy3 = np.amax(accuracy_array3)
    feat_ind_max_acc3 = feat_index_arry[np.argmax(accuracy_array3)]
    print('Random Forest accuracy = %s' % max_accuracy3)
    print('Random Forest Num of features at max accuracy = %s' % feat_ind_max_acc3)

    # Random forest plot
    randomF_plt = plt.plot(feat_index_arry, accuracy_array3, 'y')
    randomF_max_point = plt.plot(feat_ind_max_acc3, max_accuracy3, 'kx')
    plt.hlines(max_accuracy3, 0, feat_ind_max_acc3, linestyles='dashed', colors='k', linewidth=0.5)
    plt.vlines(feat_ind_max_acc3, 0, max_accuracy3, linestyles='dashed', colors='k', linewidth=0.5)

    return randomF_plt


"""
------------------------------Multithreading to speed up processing time------------------------
"""
def thread_func(count, selected_feat, train_act, train_label, test_act, test_label):
    """
    # select 'm'- number of feature(s) for each iteration using correlation based feature selection technique (selectKbest)
    selected_feat = SelectKBest(chi2, k=count)

    # update feature index for each iteration
    feat_index_arry[count - 1] = count
    """

    svm_class(count, selected_feat, train_act, train_label, test_act, test_label)
    #elif algo == 2:
    knn_class(count, selected_feat, train_act, train_label, test_act, test_label)
    #elif algo == 3:
    #rand_forest_class(count, selected_feat, train_act, train_label, test_act, test_label)
    """
    else:
        print("----error! invalid system argument found")
        sys.exit()
    """


"""
-------------------------------------------------------------------------------------------------
"""
# this loop manages iterations for the KFold cross validation
for train_index, test_index in kf.split(FEATURES_norm):
    # assign training and test data after kFold data split
    train_act, test_act = FEATURES_norm.iloc[train_index], FEATURES_norm.iloc[test_index]
    train_label, test_label = act_label.iloc[train_index], act_label.iloc[test_index]
    print("------------------for cross validation %s-----------------------------------" % kfold_count)
    threads = []

    sgd_class(train_act, train_label, test_act, test_label)
    #sys.exit()
    # this loop manages iterations for number of selected features
    for m in range(1, lengthOfFeatures):
        print("--------------------------------------------------------------------------")
        progress = 100.0*m/lengthOfFeatures
        pretty_print('Count= %s' % m, progress)
        # print('Count= %s' % m)

        # select 'm'- number of feature(s) for each iteration using correlation based feature selection technique (selectKbest)
        selected_feat = SelectKBest(chi2, k=m)

        # update feature index for each iteration
        feat_index_arry[m - 1] = m
        """
        # create threads to speed up process
        
        t = Thread(target=thread_func, args=(m, selected_feat, train_act, train_label, test_act, test_label))
        #my_queue.put(t)
        threads.append(t)
        #t.start()

    for x in threads:
        x.start()
        #time.sleep(0.2)
    # wait till all threads have run

    for m in range(1, lengthOfFeatures):
        print("--------------------------------------------------------------------------")
        progress = 100.0*m/lengthOfFeatures
        pretty_print('Count= %s' % m, progress)
        # print('Count= %s' % m)

        # select 'm'- number of feature(s) for each iteration using correlation based feature selection technique (selectKbest)
        selected_feat = SelectKBest(chi2, k=m)
        rand_forest_class(m, selected_feat, train_act, train_label, test_act, test_label)

    for t in threads:
        t.join()
        """

        # svm
        svm_class(m, selected_feat, train_act, train_label, test_act, test_label)

        # knn
        knn_class(m, selected_feat, train_act, train_label, test_act, test_label)

        #random forest
        rand_forest_class(m, selected_feat, train_act, train_label, test_act, test_label)

    print("-------------------------------------------------------------------------------")
    # Plot
    plt.figure(kfold_count)
    print("Total time to achieve final results = ... %s seconds..." % (time.time() - start_time))
    print("Plotting results.........................")
    plt.xlim(0, 100)
    plt.ylim(0, 1.2)
    plt.title('Accuracy vs Features')

    svm_plotting = svm_plot()
    knn_plotting = knn_plot()
    randomF_plotting = randomF_plot()

    plt.legend([svm_plotting[0], knn_plotting[0], randomF_plotting[0]], ['SVM', 'KNN', 'RandomF'])
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.grid(linestyle='dotted', linewidth=0.3)
    kfold_count += 1
plt.show()



print("Program Ended... %s seconds..." % (time.time() - start_time))