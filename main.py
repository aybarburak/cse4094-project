# Marmara University, Computer Engineering Department
# Natural Language Processing, Machine Learning Project
# Burak Aybar 150112001 & Farid Yagubbayli 150113901

import os.path

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm.classes import LinearSVC

from batch import MultinomialNB_tester, Logit_tester, LinearSVC_tester
from util import output_to_file, file_to_features_labels, trainer, tester
import gc

# Paths to data files
FILE_TRAIN = "train.txt"
FILE_TEST = "test.txt"

# Feature sets
featureSets = [
    [(-1, 0), (-1, 1), (0, 0), (0, 1)],
    [(-2, 0), (-2, 1), (-1, 0), (-1, 1), (0, 0), (0, 1)],
    [(-1, 1), (0, 1)],
    [(-2, 1), (-1, 1)],
    [(-2, 1), (-1, 1), (0, 0), (0, 1)]
]

# Execute in ordinary form, without threads and custom parameters
def ordinary_execution():

    # List of classifiers
    clfs = [MultinomialNB(), LogisticRegression(), LinearSVC()]

    for clf in clfs:
        print("Classifier: " + str(clf))

        for i in range(len(featureSets)):
            print("Feature Set #" + str(i))
            gc.collect()    # For efficient usage of the memory

            # Get necessary data sets
            train_features, train_labels = file_to_features_labels(FILE_TRAIN, featureSets[i])
            test_features, test_labels = file_to_features_labels(FILE_TEST, featureSets[i])

            # Train classifier
            vectorizer, clf = trainer(train_features, train_labels, clf)
            # Test classifier
            results = tester(test_features, test_labels, vectorizer, clf)

            print("Results: " + str(results))
        print("\n")

# Execute in threaded form, with a set of custom parameters
# @classifier_tester: Instance of classifier tester from batch.py
# @classifier_label: Used when writing to the output file
def threaded_execution(classifier_tester, classifier_label):
    for i in range(len(featureSets)):
        gc.collect()    # For efficient usage of the memory

        print("Feature set: #" + str(i))
        results = classifier_tester( featureSets[i])

        #Write results to file
        output_to_file(classifier_label, i, results)

if __name__ == "__main__":

    # If output file exists, remove it
    if os.path.exists("experiment_results.txt"):
        os.remove("experiment_results.txt")

    execution_type = "threaded" #or threaded

    if execution_type == "ordinary":
        ordinary_execution()
    elif execution_type == "threaded":
        threaded_execution(MultinomialNB_tester, "Multinomial Naive Bayes")




