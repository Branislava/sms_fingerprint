#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

def batch_classify(X_train, Y_train, X_test, Y_test, verbose = True):

    df_results = pd.DataFrame(data=np.zeros(shape=(no_classifiers, 4)), columns=['classifier', 'train_score', 'test_score', 'training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        df_results.loc[count, 'classifier'] = key
        df_results.loc[count, 'train_score'] = train_score
        df_results.loc[count, 'test_score'] = test_score
        df_results.loc[count, 'training_time'] = t_diff
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key, f=t_diff))
        count += 1
    return df_results

def modelfit(alg, X_train, y_train, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):

    # fit the algorithm on the data
    alg.fit(X_train, y_train)
        
    # predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:, 1]

    # perform cross-validation:
    if performCV:
        cv_score = model_selection.cross_val_score(alg, X_train, y_train, cv=cv_folds, scoring='roc_auc')
    
    # print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
        
    # print feature importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)[:15]
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('python3.5 main.py dataset/data-sms-reg.csv')
        print('python3.5 main.py dataset/data-sms-form.csv')
        exit(1)
    
    df = pd.read_csv(sys.argv[1])
    y = df['class']
    X = df.ix[:, df.columns != 'class']

    # train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print('Before feature selection ', np.shape(X_train), np.shape(y_train))
    
    # creating model
    clf = GradientBoostingClassifier()
    
    # perform feature selection
    predictors = np.array(X.columns.values)
    
    # fitting model
    clf.fit(X_train, y_train)
    modelfit(clf, X, y, predictors)
    
    # predicting labels
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    # trying out different classifiers
    dict_classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Linear SVM": LinearSVC(C=0.1, penalty='l1', dual=False),
        "Gradient Boosting Classifier": clf,
        "Random Forest": RandomForestClassifier(n_estimators = 18),
        "Neural Net": MLPClassifier(alpha = 1),
    }
    no_classifiers = len(dict_classifiers.keys())
    
    df_results = batch_classify(X_train, y_train, X_test, y_test)
    print(df_results.sort_values(by='test_score', ascending=False))
