# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:39:03 2019

@author: Alexandre
"""

import random
import data_extraction
import naive_bias
import sklearn_models


###############################################################################
#To have a result
###############################################################################
max_data = 12500
nb_folds = 10 #How many folds you want
data_part = 0.01 #partition (from 0 to 1) of the dataset that you want to use
kfolds = 6 #Number (of nb_folds) of kfolds used for the present result
folds = int( data_part * max_data / nb_folds )
data_range = random.randint(0,max_data-nb_folds*folds)
nb_hf_1_grams = 500
nb_hf_2_grams = 2250
nb_hf_3_grams = 4500
nb_hf_4_grams = 2250

#(X_training,Y_training,X_validation,Y_validation) = data_extraction.ExtractDataTraining(data_part,data_range,kfolds,nb_folds,folds,nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams)
#F1 = naive_bias.Accuracy(X_training,Y_training,X_validation,Y_validation)


###############################################################################
#To have a result with a cross validation
###############################################################################
data_part = 0.01 #partition (from 0 to 1) of the dataset that you want to use
#F1 = naive_bias.CrossValidation(data_part)


###############################################################################
#To have a result from the testing data
###############################################################################
nb_hf_1_grams = 500
nb_hf_2_grams = 2250
nb_hf_3_grams = 4500
nb_hf_4_grams = 2250

#(X_training,Y_training,X_testing,data_name) = data_extraction.ExtractDataTesting(nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams)
#naive_bias.Submission(X_training,Y_training,X_testing,data_name)


###############################################################################
#To have a result with the SKLearn models
###############################################################################
max_data = 12500
nb_folds = 10 #How many folds you want
data_part = 0.01 #partition (from 0 to 1) of the dataset that you want to use
kfolds = 6 #Number (of nb_folds) of kfolds used for the present result
folds = int( data_part * max_data / nb_folds )
data_range = random.randint(0,max_data-nb_folds*folds)
nb_hf_1_grams = 500
nb_hf_2_grams = 2250
nb_hf_3_grams = 4500
nb_hf_4_grams = 2250
model = 1 # SKLearn model : Logistic Regression
#model = 2 # SKLearn model : Descision Tree Classifier

#(X_training,Y_training,X_validation,Y_validation) = data_extraction.ExtractDataTraining(data_part,data_range,kfolds,nb_folds,folds,nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams)
#F1 = naive_bias.Accuracy(X_training,Y_training,X_validation,Y_validation,model)



###############################################################################
#To have a result from the training data but with SKLearn data extraction
###############################################################################
#sklearn_models.SKLearn_extraction()
