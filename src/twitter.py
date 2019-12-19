"""
Outline Author      : Yi-Chieh Wu, Sriram Sankararman
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'r') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list

        #for every tweet in the text file
        for tweet in fid:
            unique = extract_words(tweet)
            for word in unique:
                #check if the word is not in the dict
                if word not in word_list:
                    #add the word to the dictionary
                    word_list[word] = len(word_list)

        pass
        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'r'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'r') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix

        i = 0
        for line in fid:
            for word in extract_words(line):
                #toggle the bit if the word is in the tweet
                feature_matrix[i][word_list[word]] = 1
            i = i + 1

        pass
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc'       
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance

    score = 0
    if (metric == "f1_score"):
        #compute f1 score
        score = metrics.f1_score(y_true, y_label)
    elif (metric == "auroc"):
        #compute auroc score
        score = metrics.roc_auc_score(y_true, y_label)
    else:
        #compute accuracy score
        score = metrics.accuracy_score(y_true, y_label)

    return score
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance

    p_scores = []

    for train_index, test_index in kf.split(X, y):
        #collect the split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #train the classifier
        clf.fit(X_train, y_train)

        #predict using development
        y_pred = clf.decision_function(X_test)

        #report perfomance
        p_scores.append(performance(y_test, y_pred, metric))

    #take average of the scores
    return np.mean(p_scores)

    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    #print ('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2: select optimal hyperparameter using cross-validation

    #find score for each value of hyperparameter
    hyp_score = []

    for C in C_range:
        #create classifier with given hyperparameter
        clf = SVC(C=C, kernel='linear')

        #find the cross validation scores
        score = cv_performance(clf, X, y, kf, metric=metric)
        hyp_score.append(score)
    
    #print the list of scores for the homework
    print(hyp_score)

    #find the hyperparameter with the greatest score
    index = hyp_score.index(max(hyp_score))

    print ('\t-- Hyperparameter Selection based on ' + str(metric) + ' : %.1f' % (C_range[index]))

    return C_range[index]
    ### ========== TODO : END ========== ###



def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 3: return performance on test data by first computing predictions and then calling performance

    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric=metric)

    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    print("Parsing tweets ...")

    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc"]
    
    ### ========== TODO : START ========== ###

    print("\t-- Dimensions of Feature Vector array: %s" % (X.shape, ))

    # part 1: split data into training (training + cross-validation) and testing set

    # training = first 560 tweets, testing = last 70 tweets per the instructions
    print("Splitting tweets ...")
    X_train = X[0:560]
    X_test = X[560:630]
    y_train = y[0:560]
    y_test = y[560:630]

    
    # part 2: create stratified folds (5-fold CV)

    skf = StratifiedKFold(n_splits=5)

    # part 2: for each metric, select optimal hyperparameter for linear-kernel SVM using CV

    print("Selecting hyperparameters ...")

    hyp_dict = {}
    for metric in metric_list:
        #find best hyperparameter
        best_hyp = select_param_linear(X_train, y_train, skf, metric=metric)
        hyp_dict[metric] = best_hyp
        
    # part 3: train linear-kernel SVMs with selected hyperparameters

    print("Training linear-kernel SVMs ...")

    #create linear SVM classifier for accuracy with selected hyperparameter
    acc_clf = SVC(C=hyp_dict["accuracy"], kernel='linear')
    #train classifier
    acc_clf.fit(X_train, y_train)

    #create linear SVM classifier for f1 score with selected hyperparameter
    f1_clf = SVC(C=hyp_dict["f1_score"], kernel='linear')
    #train classifier
    f1_clf.fit(X_train, y_train)

    #create linear SVM classifier for auroc with selected hyperparameter
    auroc_clf = SVC(C=hyp_dict["auroc"], kernel='linear')
    #train classifier
    auroc_clf.fit(X_train, y_train)
    
    # part 3: report performance on test data

    print("Testing classifiers ...")

    # use performance_test function
    test_scores = {}
    test_scores["accuracy"] = performance_test(acc_clf, X_test, y_test, metric="accuracy")
    test_scores["auroc"] = performance_test(auroc_clf, X_test, y_test, metric="auroc")
    test_scores["f1_score"] = performance_test(f1_clf, X_test, y_test, metric="f1_score")

    #print performance
    print("Test Performance:")
    for metric in metric_list:
        print("\t-- %s performance: %.4f" % (metric, test_scores[metric]))

    print("Done.")
    
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
