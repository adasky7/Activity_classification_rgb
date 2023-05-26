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
from sklearn import preprocessing, model_selection, svm, neighbors
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from Features_extended import compute_features
import time, sys
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, chi2, f_classif, SelectFromModel, mutual_info_classif
from sklearn.pipeline import Pipeline, make_pipeline
import traceback
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneGroupOut, KFold, GroupKFold
from skrebate import ReliefF

# set program time
start_time = time.time()
start_time_local = time.time()


# control system process time
def pretty_print(message, progress, reset=False):
    global start_time_local

    if (reset):
        start_time_local = time.time()

    current_time = time.time() - start_time
    current_time_local = time.time() - start_time_local
    print("G[%.3f] L[%.3f] [%.0f%%]: %s" % (current_time, current_time_local, progress, message))

#Extract train and test data from computed features
def Ext_features(datafile):

    ex_features = compute_features(datafile)
    # ex_features.to_csv('myresults.csv', sep=',')
    data = ex_features.drop('label', 1)  # drop the column of activity label
    FEATURES = data.fillna(method='ffill')
    # print(data.isnull().any())
    act_label = ex_features.label
    #print('Completed feature extraction')

    """
    Normalize features using Standardization or z-score
    """
    # Min-Max normalization
    FEATURES_norm = (FEATURES - FEATURES.min()) / (FEATURES.max() - FEATURES.min())
    FEATURES_norm = FEATURES_norm.fillna(0)
    #print("Completed feature normalization")


    return FEATURES_norm, act_label

"""
---------------------Classification functions-----------------------------------------------------
"""

# setup svm classifier
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


# setup knn classifier
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
        dontdoabsolutelyanythingelseplease = 1

#Ensemble of classifiers
def en_class(value, selected_feat, trainA, trainL, testA, testL):
    global accuracy_array4

    pretty_print('Classifier: Ensemble pipeline with selected feature', progress, True)
    #Classifier ensemble with selected feature
    en_pipe = make_pipeline(selected_feat, en_classifier)
    en_pipe.fit(trainA, trainL)
    _accuracy4 = en_pipe.score(testA,testL)
    pretty_print('En_Class %s' % _accuracy4, progress)
    accuracy_array4[value - 1] = _accuracy4
    # print performance metric
    # prediction = en_pipe.predict(testA)
    # print(classification_report(testL, prediction))



"""
Import features computed
"""
file1 = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person1still.csv'
file2 = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person2still.csv'
file3 = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person3still.csv'
file4 = '/home/dadama/Dropbox/PhD personal docs/Resources/Cornell Univ data set/person4still.csv'
trainfile = [file1, file2, file3]
testfile = [file4]

#seperate individual persons
person1, p1 = Ext_features([file1])
person2, p2 = Ext_features([file2])
person3, p3 = Ext_features([file3])
person4, p4 = Ext_features([file4])
group1 = pd.DataFrame(np.zeros((len(p1.index), 1)))
group1[0] = 1
group2 = pd.DataFrame(np.zeros((len(p2.index), 1)))
group2[0] = 2
group3 = pd.DataFrame(np.zeros((len(p3.index), 1)))
group3[0] = 3
group4 = pd.DataFrame(np.zeros((len(p4.index), 1)))
group4[0] = 4
groups = pd.concat([group1, group2, group3, group4], axis=0).reset_index(drop=True)
dataAct = pd.concat([person1, person2, person3, person4], axis=0).reset_index(drop=True)
dataLabel = pd.concat([p1, p2, p3, p4], axis=0).reset_index(drop=True)


#assign train and test data extracted from dataset
print('Training data')
train_act, train_label = Ext_features(trainfile)
print('Test data')
test_act, test_label = Ext_features(testfile)


"""
feature selection using lsvc with l1 penalty
"""
dataAll = dataAct
LabelAll = dataLabel
print('Feature selection')
"""
lsvc = svm.LinearSVC(penalty='l1', dual=False).fit(dataAct, dataLabel)
model = SelectFromModel(lsvc, prefit=True)
#train_new = model.transform(train_act)
#test_new = model.transform(test_act)
#train_new = pd.DataFrame(train_new)
#test_new = pd.DataFrame(test_new)
#print(train_new.shape)
#print(train_new)

#cross val test
#data = [file1, file2, file3, file4]
#dataAct, dataLabel = Ext_features(data)
dataAct = model.transform(dataAct)
dataAct = pd.DataFrame(dataAct)
print(dataAct.shape)
"""

"""
Test Relief-F feature selection
"""
fs = ReliefF()
#fs.fit(dataAll.as_matrix(), LabelAll.as_matrix())
print("Selected features using Relief F")
#print(fs.feature_importances_)


"""
Variable declaration used in feature selection and classification
"""
feat_index_arry = np.zeros(len(train_act.columns))
accuracy_array = np.zeros(len(train_act.columns))
accuracy_array2 = np.zeros(len(train_act.columns))
accuracy_array3 = np.zeros(len(train_act.columns))
accuracy_array4 = np.zeros(len(train_act.columns))

classifier = svm.SVC(kernel='linear', probability=True)
classifier2 = neighbors.KNeighborsClassifier(n_jobs=-1)
classifier3 = RandomForestClassifier(n_jobs=-1)
en_classifier = VotingClassifier(estimators=[('svm', classifier), ('knn', classifier2), ('rf', classifier3)], voting='soft', n_jobs=-1)

lengthOfFeatures = len(train_act.columns) + 1


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
-------------------------------------------------------------------------------------------------
"""

# this loop manages iterations for number of selected features
for m in range(1, 2):
    print("--------------------------------------------------------------------------")
    progress = 100.0 * m / lengthOfFeatures
    pretty_print('Count= %s' % m, progress)
    # print('Count= %s' % m)

    # select 'm'- number of feature(s) for each iteration using correlation based feature selection technique (selectKbest)
    #selected_feat = SelectKBest(mutual_info_classif, k='all')
    selected_feat = SelectPercentile(mutual_info_classif, percentile=90)
    #print ((selected_feat.fit(dataAct, dataLabel)).scores_)
    # print(selected_feat.get_support(indices=True))
    # update feature index for each iteration
    feat_index_arry[m - 1] = m

    # svm
    #svm_class(m, selected_feat, train_act, train_label, test_act, test_label)

    # knn
    #knn_class(m, selected_feat, train_act, train_label, test_act, test_label)

    # random forest
    #rand_forest_class(m, selected_feat, train_act, train_label, test_act, test_label)

    #classifier ensemble
    #en_class(m, selected_feat, train_new, train_label, test_new, test_label)
    #en_class(m, selected_feat, train_act, train_label, test_act, test_label)
    cv = GroupKFold(n_splits=4)
    en_pipe = make_pipeline(selected_feat, en_classifier)
    print("Using selected features")
    cv_predict = cross_val_predict(en_pipe, dataAct, dataLabel, cv=cv, groups=groups)
    #cv_score = cross_val_score(en_pipe, train_act, train_label, cv=cv)
    print("Ensemble cross validation report (selected features) %s" % classification_report(dataLabel, cv_predict, digits=4))
    cnf_matrix = confusion_matrix(dataLabel, cv_predict)
    cnf_matrix = cnf_matrix.astype(np.float) / cnf_matrix.sum(axis=1)
    print("Confusion matrix (selected features) %s" % (cnf_matrix*100))

    print("using all features")
    cv_predict = cross_val_predict(en_classifier, dataAll, LabelAll, cv=cv, groups=groups)
    print("Ensemble cross validation report (All features) %s" % classification_report(LabelAll, cv_predict,
                                                                                            digits=4))
    print("Confusion matrix (All features) %s" % confusion_matrix(LabelAll, cv_predict))

"""
print("-------------------------------------------------------------------------------")
# Plot
#plt.figure(kfold_count)
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
#kfold_count += 1
plt.show()

print("Program Ended... %s seconds..." % (time.time() - start_time))
"""