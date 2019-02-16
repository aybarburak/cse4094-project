# Marmara University, Computer Engineering Department
# Natural Language Processing, Machine Learning Project
# Burak Aybar 150112001 & Farid Yagubbayli 150113901

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support

# Read file and make a list of sentences (Structure the data)
# @file_path: Path to file that will be readen
# @return: List of lists. Inner list is like => "word", "POS Tag", "Chunk Tag / Class Label"
def fileToSet(file_path):
    # Read training data, store as sentences
    res_set = []
    with open(file_path) as f:
        sentence = []
        for line in f:
            # Get elements of line that have blank space between each other
            line_elements = line.strip('\n').split(' ')

            if len(line_elements) < 3:  # Empty line, new Sentence will begin
                res_set.append(sentence)
                sentence = []
            else:  # Part of the current Sentence
                sentence.append(line_elements)
    return res_set


# Data set to feature set, makes easier to build different feature sets
# @dataSet: Set to work on
# @featurePos: List of tuples. Where,
#       1st item of tuple => How many words before/after
#       2nd item of tuple => Word itself (0), POS Tag (1) or Chunk Tag (2)
# @return1: List of dictionaries.
#       Each dictionary is set of features with keys of f#, where # is number
# @return2: List of labels (or categories)
def featureSetBuilder(dataSet, featurePos):
    features = []
    labels = []

    for sentence in dataSet:
        for i in range(0, len(sentence)):
            featureRecord = {}
            fcnt = 0
            for fPos in featurePos:
                # If index is negative, feature is ''
                featureVal = '' if i < abs(fPos[0]) else sentence[i + fPos[0]][fPos[1]]
                # Example - {'f1': 'hello', 'f2': 'bye', ......}
                featureRecord['f' + str(fcnt)] = featureVal
                fcnt += 1

            features.append(featureRecord)

            # Also get the label
            labels.append(sentence[i][2])
    return features, labels


# Train on the features with given classifier and feature selections
# @features: List of dictionaries. Each dictionary consists from a set of features
# @labels: Label of each feature set
# @clf: Classifier
# @return1: DictVectorizer object that was used to vectorize features
# @return2: Trained classifier object
def trainer(features, labels, clf):
    dict_vectorizer = DictVectorizer(sparse=True)
    X = dict_vectorizer.fit_transform(features)

    clf.fit(X, labels)
    return dict_vectorizer, clf

# Test the trained classifier and give resulting measurements
# @features: List of dictionaries. Each dictionary consists from a set of features
# @labels: Label of each feature set
# @vectorizer: An instance of DictVectorizer
# @clf: Classifier
# @return1: Percentage of correctly classified elements
def tester(features, labels, vectorizer, clf):
    X = vectorizer.transform(features)
    predictions = clf.predict(X)
    #mean = np.mean(predictions == labels)

    res = precision_recall_fscore_support(labels, predictions, average='micro')
    res = (round(res[0], 3), round(res[1], 3), round(res[2], 3))

    return res

# Read file and make two data structures which are holding 'features' and 'labels'
# @file: Path to file
# @featurePos: Positions of feature values in respect to current word
def file_to_features_labels(file, featurePos):
    training_set = fileToSet(file)
    features, labels = featureSetBuilder(training_set, featurePos)
    return features, labels

def output_to_file(methodName, featureSetIndex, results):

    file = open("experiment_results.txt", "a")
    file.write(methodName + " : " + str(featureSetIndex) + "\n")

    for res in results:
        file.write(str(res))
        file.write("\n")


    file.write("\n\n")

    file.flush()
    file.close()