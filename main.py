#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# Run classifier with cross-validation and plot ROC curves
def plot_roc(classifier, X, y, k, predictors):
    cv = StratifiedKFold(n_splits=k)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])

        if i == 0:
            feat_imp = pd.Series(classifier.feature_importances_, predictors).sort_values(ascending=False)[:15]
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.tight_layout()
            plt.show()

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('python3.5 main.py dataset/data-sms-reg.csv')
        print('python3.5 main.py dataset/data-sms-form.csv')
        exit(1)

    # dataset
    df = pd.read_csv(sys.argv[1])

    # samples and labels
    y = df['class']
    X = df.ix[:, df.columns != 'class']

    # creating classification model
    classifier = GradientBoostingClassifier(learning_rate=0.1, n_estimators=160, min_samples_split=10, min_samples_leaf=30, max_depth=9, max_features=11, subsample=0.8, random_state=10)

    # number of folds
    k = 5

    # CV score
    scores = model_selection.cross_val_score(classifier, X, y, cv=k, scoring='roc_auc')
    print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(scores), np.std(scores), np.min(scores), np.max(scores)))

    # plotting ROC curve
    plot_roc(classifier, X.as_matrix(), np.array(y), k, np.array(X.columns.values))
