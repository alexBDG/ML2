# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:10:03 2019

@author: Alexandre
"""

import numpy as np
import random
import naive_bias
import data_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

###############################################################################
#SKLearn data extraction with LogisticRegression model
###############################################################################
def SKLearn_extraction():
    max_data = 12500
    nb_folds = 10
    data_part = 1
    kfolds = 6
    folds = int( data_part * max_data / nb_folds )
    data_range = random.randint(0,max_data-nb_folds*folds)
    nb_hf_1_grams = int(500*data_part)
    nb_hf_2_grams = int(2250*data_part)
    nb_hf_3_grams = int(4500*data_part)
    nb_hf_4_grams = int(2250*data_part)
    
    (data_train,Y_train,data_test,Y_test) = data_extraction.ExtractDataTraining(data_part,data_range,kfolds,nb_folds,folds,nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams,True)
    
    #BAG OF WORDS
    count_vect = CountVectorizer().fit(data_train)
    X_train_counts = count_vect.transform(data_train)
    X_test_counts = count_vect.transform(data_test)
    
    #BETTER FEATURES
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    
    #NORMALIZATION
    normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
    X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
    X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)
    
    #THE MODEL
    clf = LogisticRegression(random_state=0,penalty='l2').fit(X_train_normalized,Y_train)
    
    #PREDICT AND EVALUATION
    Y_pred = clf.predict(X_test_normalized)
    print(metrics.classification_report(Y_test,Y_pred))


###############################################################################
#Compute the confusion matrix of the model Logistic Regression from SKLearn
###############################################################################              
def ConfusionMatrixLogReg(X_training,Y_training,X_testing,Y_testing):
    p=len(Y_testing)
    Y_prediction = np.zeros(p)
    MC = np.zeros((2,2),dtype=int)

    clf = LogisticRegression(random_state=0,penalty='l2').fit(X_training,Y_training)
    Y_prediction = clf.predict(X_testing)
    
    for k in range(p):
        MC += naive_bias.AddMatrix(Y_testing[k],Y_prediction[k])
        
    return MC    


###############################################################################
#Compute the confusion matrix of the model Decision Tree Classifier from SKLearn
###############################################################################              
def ConfusionMatrixTree(X_training,Y_training,X_testing,Y_testing):
    p=len(Y_testing)
    Y_prediction = np.zeros(p)
    MC = np.zeros((2,2),dtype=int)

    clf = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0).fit(X_training,Y_training)
    Y_prediction = clf.predict(X_testing)
    
    for k in range(p):
        MC += naive_bias.AddMatrix(Y_testing[k],Y_prediction[k])
        
    return MC    
