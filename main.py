#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

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

    # creating classification model
    classifier = GradientBoostingClassifier(learning_rate=0.1, n_estimators=160, min_samples_split=10, min_samples_leaf=30, max_depth=9, max_features=11, subsample=0.8, random_state=10)

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
        f1_pos.append(f1_score(y[test], y_pred, pos_label=pos_label))
        prec_pos.append(precision_score(y[test], y_pred, pos_label=pos_label))
        rec_pos.append(recall_score(y[test], y_pred, pos_label=pos_label))
        
        # evaluating towards negative class
        pos_label = 1
        f1_neg.append(f1_score(y[test], y_pred, pos_label=pos_label))
        prec_neg.append(precision_score(y[test], y_pred, pos_label=pos_label))
        rec_neg.append(recall_score(y[test], y_pred, pos_label=pos_label))

        if feat_plot:
            feat_plot = False
            feat_imp = pd.Series(classifier.feature_importances_, predictors).sort_values(ascending=False)[:15]
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.tight_layout()
            plt.show()
    
    print('Accuracy %.3f' % np.mean(acc))
    
    print('Positive class (pos_label = 0)')
    print('F-score %.3f' % np.mean(f1_pos))
    print('Precision %.3f' % np.mean(prec_pos))
    print('Recall %.3f' % np.mean(rec_pos))

    print('Negative class (pos_label = 1)')
    print('F-score %.3f' % np.mean(f1_neg))
    print('Precision %.3f' % np.mean(prec_neg))
    print('Recall %.3f' % np.mean(rec_neg))