#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.neighbors import KNeighborsClassifier

from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('python3.5 main.py dataset/data-sms-reg.csv')
        print('python3.5 main.py dataset/data-sms-form.csv')
        exit(1)

    # dataset
    df = pd.read_csv(sys.argv[1])

    # samples and labels
    predictors = df.ix[:, df.columns != 'class'].columns.values
    y = np.array(df['class'])
    X = np.array(df.ix[:, df.columns != 'class'])
    
    names = [
        "Baseline",
        #"Nearest Neighbors (k=13)", 
        #"Nearest Neighbors (k=5)", 
        #"Nearest Neighbors (k=50)", 
        "Linear SVM (C=0.025)", 
        "Linear SVM (C=1)", 
        "RBF SVM", 
        #"Gaussian Process",
        #"Decision Tree", 
        #"Random Forest", 
         "Neural Net", 
        #"AdaBoost",
        #"Naive Bayes", 
        #"QDA", 
         "Gradient Boosting Classifier"]

    classifiers = [
        DummyClassifier(strategy='most_frequent',random_state=0),
	#KNeighborsClassifier(3),
        #KNeighborsClassifier(5),
        SVC(kernel="linear", C=0.025),
        SVC(kernel="linear", C=1),
        SVC(kernel="rbf", gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        #DecisionTreeClassifier(max_depth=5),
        #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        #AdaBoostClassifier(),
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
        GradientBoostingClassifier(learning_rate=0.1, n_estimators=160, min_samples_split=10, min_samples_leaf=30, max_depth=9, max_features=11, subsample=0.8, random_state=10)]


    print('Classifier,Accuracy,Precision (positive class),Recall (positive class),F-score (positive class),Precision (negative class),Recall (negative class),F-score (negative class)')
    for name, classifier in zip(names, classifiers):

        # performing cross validation
        cv = StratifiedKFold(n_splits=5)
        
        feat_plot = True
        acc = []
        f1_pos, prec_pos, rec_pos = [], [], []
        f1_neg, prec_neg, rec_neg = [], [], []
        
        for train, test in cv.split(X, y):
            
            y_pred = classifier.fit(X[train], y[train]).predict(X[test])
            acc.append(accuracy_score(y[test], y_pred))
            
            # evaluating towards positive class
            pos_label = 0
            f1_pos.append(f1_score(y[test], y_pred, pos_label=pos_label, average='binary'))
            prec_pos.append(precision_score(y[test], y_pred, pos_label=pos_label, average='binary'))
            rec_pos.append(recall_score(y[test], y_pred, pos_label=pos_label, average='binary'))
            
            # evaluating towards negative class
            pos_label = 1
            f1_neg.append(f1_score(y[test], y_pred, pos_label=pos_label, average='binary'))
            prec_neg.append(precision_score(y[test], y_pred, pos_label=pos_label, average='binary'))
            rec_neg.append(recall_score(y[test], y_pred, pos_label=pos_label, average='binary'))

            '''
            if feat_plot:
                feat_plot = False
                feat_imp = pd.Series(classifier.feature_importances_, predictors).sort_values(ascending=False)[:15]
                feat_imp.plot(kind='bar', title='Feature Importances')
                plt.ylabel('Feature Importance Score')
                plt.tight_layout()
                plt.show()
            '''
        
        print('%s & $%.3f$  & $%.3f$  & $%.3f$  & $%.3f$  & $%.3f$  & $%.3f$  & $%.3f$ \\\\\\hline' % (name, np.mean(acc), np.mean(prec_pos), np.mean(rec_pos), np.mean(f1_pos), np.mean(prec_neg), np.mean(rec_neg), np.mean(f1_neg)))
        sys.stdout.flush()
