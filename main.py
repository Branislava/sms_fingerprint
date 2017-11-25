import sys
import random
import numpy as np
from features_extraction.dataset import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python main.py path_to_xml_file')
        exit(1)

    # create dataset frame
    dataset = Dataset(filename=sys.argv[1])

    # training set
    N = len(dataset.data)
    training_size = int(0.8 * N)
    indexes = np.array(range(N))
    random.shuffle(indexes)
    training_indexes = indexes[:training_size]
    test_indexes = indexes[training_size:]

    TMP = random.sample(range(training_size), 1)[0]
    print(dataset.data['body'].iloc[TMP])
    print(dataset.data.iloc[TMP])

    # training set - before feature selection
    X = dataset.data.drop(['address', 'type', 'body'], 1, inplace=False).iloc[training_indexes, :]
    Y_train = dataset.data['type'].iloc[training_indexes]

    # perform feature selection
    print('...selecting top features')
    feature_names = X.columns.values
    select_k_best_classifier = SelectKBest(f_classif, k='all')
    select_k_best_classifier.fit_transform(X, Y_train)
    mask = select_k_best_classifier.get_support()
    new_features = [feature for bool, feature in zip(mask, feature_names) if bool]
    print('Selected %d features: %s' % (len(new_features), ', '.join(new_features)))

    # new training set
    X_train = dataset.data[new_features].iloc[training_indexes, :]

    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Now apply the transformations to the data:
    x = dataset.data[new_features].iloc[test_indexes, :]
    X_test = scaler.transform(x)
    Y_test = dataset.data['type'].iloc[test_indexes]

    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']
    max_iter = 1000
    random_states = [0, 1]
    tols = [0.1, 0.01, 0.001, 0.0001]
    warm_starts = [True, False]

    for activation in activations:
        for solver in solvers:
            for random_state in random_states:
                for tol in tols:
                    for warm_start in warm_starts:
                        print('Activation %s, Solver %s, Random state %d, tols %f, Warm start %r'
                              % (activation, solver, random_state, tol, warm_start))

                        # multi-layer perceptron classifier
                        mlp = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes=(30, 30, 30),
                                            random_state=random_state, activation=activation,
                                            tol=tol, warm_start=warm_start)

                        # fitting data
                        mlp.fit(X_train, Y_train)

                        # predicting labels of unseen samples
                        predictions = mlp.predict(X_test)
                        print(confusion_matrix(Y_test, predictions))
                        print(classification_report(Y_test, predictions))
