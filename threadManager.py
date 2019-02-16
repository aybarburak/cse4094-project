# Marmara University, Computer Engineering Department
# Natural Language Processing, Machine Learning Project
# Burak Aybar 150112001 & Farid Yagubbayli 150113901


# >>>>>> Important note about implementation <<<<<<<<<
# We have implemented code in this file in an Object Oriented way
# But due to Python's problems in <Pickle> we haven't been able to call object's methods
# by "map" function. Our research has shown that the procedure for solving this problem
# will add additional complexity to project. Also there are even more complicated solutions
# with 3rd party libraries.
# As a result we have implemented this code in modular way, instead of Object Oriented way.
# We hope that Python will solve it's own problems in near feature
# Farid Yagubbayli / Burak Aybar, 22.11.2016

from multiprocessing import Pool
import multiprocessing
from util import trainer, tester

# Data sets
global train_f
global train_l
global test_f
global test_l
global num_threads

# Used to set global variables
# @trainf: Feature set of training data
# @trainl: Label set of training data
# @testf: Feature set of test data
# @testl: Label set of training data
# @nt: Number of threads that will be used for execution
def tm_init(trainf, trainl, testf, testl, nt):
    global train_f
    global train_l
    global test_f
    global test_l
    global num_threads

    train_f = trainf
    train_l = trainl
    test_f = testf
    test_l = testl
    num_threads = nt

# Train with given classifiers and test to get scores
# @clf_and_params: List of tuples,
#               where first element is classifier
#               and second element is dictionary of parameters of classifier
# @return: Dictionary of given parameters and their scores
def tm_train_and_test( clf_and_params):
    global train_f
    global train_l
    global test_f
    global test_l
    global num_thread

    clf = clf_and_params[0]
    params = clf_and_params[1]
    me = multiprocessing.current_process()
    print("An experiment has started", me)

    vectorizer, clf = trainer(train_f, train_l, clf)
    results = tester(test_f, test_l, vectorizer, clf)

    print("Ended", me)
    return {"parameters": params, "results": results}

# Pool maker and manager
# @clfs: List of tuples,
#               where first element is classifier
#               and second element is dictionary of parameters of classifier
# @return: List of dictionaries where each has a parameter list of its classifier
#               and resulting scores
def tm_execute(clfs):
    print("Total experiments: ", len(clfs))
    global num_threads
    pool = Pool(num_threads)
    # Each element of 'clfs' will be executed separately using 'train_and_test'
    # Results will be collected into list
    # Warning for feature considerations!!! - Results may not be in the same order as input list
    results = pool.map(tm_train_and_test, clfs)

    pool.close()
    pool.join()

    return results

