import sys
import random
import numpy as np
from features_extraction.dataset import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py path_to_xml_file')
        exit(1)

    # create dataset frame
    dataset = Dataset(filename=sys.argv[1])

    # add emoji counts
    dataset.add_emoji_count('all')

    # add emoji type counts
    dataset.add_emoji_type_count('all')

    # training set
    N = len(dataset.data)
    training_size = int(0.8 * N)
    indexes = np.array(range(N))
    random.shuffle(indexes)
    training_indexes = indexes[:training_size]
    test_indexes = indexes[training_size:]

    # training set - before feature selection
    X = dataset.data.drop(['address', 'type', 'body'], 1, inplace=False).iloc[training_indexes, :]
    Y = dataset.data['type'].iloc[training_indexes]

    # perform feature selection
    print('...selecting top features')
    feature_names = X.columns.values
    select_k_best_classifier = SelectKBest(f_classif, k=10)
    select_k_best_classifier.fit_transform(X, Y)
    mask = select_k_best_classifier.get_support()
    new_features = [feature for bool, feature in zip(mask, feature_names) if bool]
    print('Selected features: %s' % ', '.join(new_features))

    # new training set
    X_new = dataset.data[new_features].iloc[training_indexes, :]

    # test set
    x = dataset.data[new_features].iloc[test_indexes, :]
    y_true = dataset.data['type'].iloc[test_indexes]

    # param search
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_new, Y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_pred = clf.predict(x)
        print(classification_report(y_true, y_pred))
        print()

    # creating classifer
    # clf = SVC(verbose=True, C=10.0, kernel='rbf', class_weight='balanced')

    # train classifier
    # print('...training classifier')
    # clf.fit(X_new, Y)

    # predict new labels
    # print('...predicting labels of unseen samples')
    # y_pred = clf.predict(x)

    # write classification report
    # print(classification_report(y_true, y_pred))