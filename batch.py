# Marmara University, Computer Engineering Department
# Natural Language Processing, Machine Learning Project
# Burak Aybar 150112001 & Farid Yagubbayli 150113901

from util import file_to_features_labels
from threadManager import tm_init, tm_execute

import numpy as np
import itertools

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm.classes import LinearSVC

# Paths to data files
FILE_TRAIN = "train.txt"
FILE_TEST = "test.txt"

# Multinomial Naive-Bayes
# Testing for 'alpha' parameter
# alpha: Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)
def MultinomialNB_tester(featureSet):

    # Get necessary data sets
    train_features, train_labels = file_to_features_labels(FILE_TRAIN, featureSet)
    test_features, test_labels = file_to_features_labels(FILE_TEST, featureSet)

    # Parameter of classifier with a range of values
    alpha = np.linspace(0, 2, 21)

    # Make list of Classifiers (for classification) and their parameters (for readable output)
    clfs = [( MultinomialNB(alpha=round(a, 3)), {'alpha': round(a, 3)} ) for a in alpha]

    # Get results from threads
    tm_init(train_features, train_labels, test_features, test_labels, 7)
    results = tm_execute(clfs)

    return results

# Logistic Regression - Tester
# @featureSet: List of dictionaries, each consisting from a set of features
def Logit_tester(featureSet):
    # Parameters that will be used for training of classifiers (each with a range of values)
    penalties = ['l1', 'l2']
    dual = [True, False]
    Cs = [1, 5, 10, 20, 50, 100]
    max_iter = [5, 10, 20, 50, 100]
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    multi_class = ['ovr', 'multinomial']

    # Generate all combinations of parameters
    all_params = [penalties, dual, Cs, max_iter, solvers, multi_class]
    all_params = list( itertools.product(*all_params))

    # Solve problems about parameters (according to Documentation).

    # FROM_DOC: The 'newton-cg', 'sag' and 'lbfgs' solvers support only l2 penalties.
    # Solution: Remove parameters with l1 penalty and one of above solvers
    all_params = [x for x in all_params if not (x[0]=='l1' and x[4]!='liblinear') ]

    # FROM_DOC: Dual formulation is only implemented for l2 penalty with liblinear solver.
    # Solution: Remove records which met the condition 'dual=True', except 'l2' & 'liblinear'.
    l2_liblinears = [x for x in all_params if x[0]=='l2' and x[4]=='liblinear']
    others = [x for x in all_params if not(x[0] == 'l2' and x[4] == 'liblinear')]
    # Remove (dual=True)s
    others = [x for x in others if x[1]==False]
    all_params = others + l2_liblinears

    # FROM_DOC: max_iter Useful only for the newton-cg, sag and lbfgs solvers
    # FROM_DOC: multi_class Works only for the 'newton-cg', 'sag' and 'lbfgs' solver.
    # Solution:
    others = [x for x in all_params if x[4] != 'liblinear']
    # max_iter and multi_class are useless, save only one instance of them
    # (same effect as removing duplicates and making fields equal to zero)
    liblinears = [x for x in all_params if x[3] == 5 and x[5] == 'ovr']
    all_params = others + liblinears

    # Remove any remaining duplicates
    all_params = list(set(all_params))

    # Sort list of tuples, for more understandable output
    all_params = sorted(all_params, key=lambda el: (el[4], el[0], el[1], el[2], el[3], el[5]))

    ##### At this point we have combinations of parameters
    ##### Above "parameter generation" part can be written more efficiently
    #####       But it will be less understandable and there will be no big improvements

    # Get necessary data sets
    train_features, train_labels = file_to_features_labels(FILE_TRAIN, featureSet)
    test_features, test_labels = file_to_features_labels(FILE_TEST, featureSet)

    # Make list of Classifiers (for classification) and their parameters (for readable output)
    clfs = []
    for p in all_params:
        pd = {'penalty': p[0], 'dual': p[1], 'C': p[2], 'max_iter': p[3], 'solver': p[4], 'multi_class': p[5]}
        clf_and_params = (LogisticRegression(**pd), pd)
        clfs.append(clf_and_params)

    # Get results from threads
    tm_init(train_features, train_labels, test_features, test_labels, 7)
    results = tm_execute(clfs)

    return results

# Linear Support Vector Machine - Tester
# @featureSet: List of dictionaries, each consisting from a set of features
def LinearSVC_tester(featureSet):
    # Parameters that will be varied from run-to-run
    Cs = [1, 10, 50, 100]
    loss = ['hinge', 'squared_hinge']
    penalty = ['l1', 'l2']
    dual = [True, False]
    multi_class = ['ovr', 'crammer_singer']
    fit_intercept = [True, False]

    # Generate all combinations of parameters
    all_params = [Cs, loss, penalty, dual, multi_class, fit_intercept]
    all_params = list(itertools.product(*all_params))

    # Solve problems about parameters (according to Documentation).

    # FROM_DOC: If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.
    only_crammer_singer = [x for x in all_params if x[4]=='crammer_singer']
    others = [x for x in all_params if x[4] != 'crammer_singer']
    only_crammer_singer = [x for x in only_crammer_singer if x[1]=='hinge' and penalty=='l1' and dual==True]
    all_params = others + only_crammer_singer

    # FROM_ERROR_Analysis: The combination of penalty='l2' and loss='hinge' is not supported
    all_params = [x for x in all_params if not (x[2]=='l2' and x[1]=='hinge')]

    # FROM_ERROR_Analysis: The combination of penalty='l1' and loss='hinge' is not supported
    all_params = [x for x in all_params if not (x[2] == 'l1' and x[1] == 'hinge')]

    # FROM_ERROR_Analysis: The combination of penalty='l1' and loss='squared_hinge' are not supported
    all_params = [x for x in all_params if not (x[2] == 'l1' and x[1] == 'squared_hinge')]

    # Remove any possible duplicates
    all_params = list(set(all_params))

    # Sort list of tuples, for more understandable output
    all_params = sorted(all_params, key=lambda el: (el[0], el[1], el[2], el[3], el[4], el[5]))


    # Get necessary data sets
    train_features, train_labels = file_to_features_labels(FILE_TRAIN, featureSet)
    test_features, test_labels = file_to_features_labels(FILE_TEST, featureSet)

    # Make list of Classifiers (for classification) and their parameters (for readable output)
    clfs = []
    for p in all_params:
        pd = {'C': p[0], 'loss': p[1], 'penalty': p[2], 'dual': p[3], 'multi_class': p[4], 'fit_intercept': p[5]}
        clf_and_params = (LinearSVC(**pd), pd)
        clfs.append(clf_and_params)


    tm_init(train_features, train_labels, test_features, test_labels, 7)
    results = tm_execute(clfs)
    return results
