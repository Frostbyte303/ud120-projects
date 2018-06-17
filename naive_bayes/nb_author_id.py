#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return (clf)

Sara = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
Chris = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

t0 = time()
clf = classify(features_train, labels_train)
print "classify training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "predict training time:", round(time()-t0, 3), "s"
print ('complete training')
print clf.score(features_test, labels_test)

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
    ### create classifier
    clf = classify(features_train, labels_train)

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = accuracy_score(labels_test, pred)
    print ("the NBA Accuracy is:",accuracy)
    return accuracy

print ('i get here')

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    print ("the submit accuracy is:",accuracy)
    return accuracy
    print (accuracy)

print ('does code reach here')

#########################################################


