# Twitter-Classification

Using Support Vector Machines (SVMs) to classify tweets as positive or negative reviews.

tweets.txt contains 630 tweets about movies. Each line in the ﬁle contains exactly one tweet, so there are 630 lines in total.

labels.txt contains the corresponding labels. If a tweet praises or recommends a movie, it is classiﬁed as a positive review
and labeled +1; otherwise it is classiﬁed as a negative review and labeled −1. These labels are ordered, i.e. the label for the
ith tweet in tweets.txt corresponds to the ith number in labels.txt.

The ﬁrst 560 tweets will be used for training and the last 70 tweets will be used for testing.

# Feature Extraction

We will use a bag-of-words model to convert each tweet into a feature vector. A bag-of-words model treats a text ﬁle as a collection
of words, disregarding word order. The ﬁrst step in building a bag-of-words model involves building a “dictionary”. A dictionary
contains all of the unique words in the text ﬁle. For this project, we will be including punctuations in the dictionary too.

extract_words(...)

- processes an input string to return a list of unique words.

extract_dictionary(...)

- uses extract_words(...) to read all unique words contained in a ﬁle into a dictionary

extract_feature_vectors(...)

- produces the bag-of-words representation of a ﬁle based on the extracted dictionary

# Hyper-parameter Selection for a Linear-Kernel SVM

Next, we will learn a classiﬁer to separate the training data into positive and negative tweets. For the classiﬁer, we will use
SVMs with the linear kernel. We will use the sklearn.svm.SVC class3 and explicitly set only three of the initialization parameters:
kernel, and C. As usual, we will use SVC.fit(X,y) to train our SVM, but in lieu of using SVC.predict(X) to make predictions, we will
use SVC.decision_function(X), which returns the (signed) distance of the samples to the separating hyperplane.

SVMs have hyperparameters that must be set by the user. For both linear kernel SVMs, we will select the hyperparameters using 5-fold
cross-validation (CV). Using 5-fold CV, we will select the hyperparameters that lead to the ‘best’ mean performance across all 5 folds.

performance(...)

- considers the following performance measures: accuracy, F1-Score, and AUROC, and returns the performance.

cv_performance(...)

- returns the mean k-fold CV performance for the performance metric passed into the function.

select_param_linear(...)

- chooses a setting for C for a linear SVM based on the training data and the speciﬁed metric.

# Test Set Performance

Apply the classiﬁer learned in the previous sections to the test data.  For each performance metric, use performance_test(...) and
the trained linear kernel SVM classiﬁer to measure performance on the test data

performance_test(...)

- returns the value of a performance measure, given the test data and a trained classiﬁer.
