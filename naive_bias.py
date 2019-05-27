# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:49:09 2019

@author: Alexandre
"""

import numpy as np
import os
import time
import csv
import matplotlib.pyplot as plt
import random
import data_extraction
import sklearn_models


###############################################################################
#Counter
###############################################################################              
def NumberOfInstance(X,theta_id):
    d=len(X[0])
    theta_i = np.zeros(d)
    for i in theta_id:
        for j in range(d):
            if X[i,j]==1:
                theta_i[j] += 1
    return theta_i


###############################################################################
#Create the theta vectors of Naives Bayes
###############################################################################              
def NaiveBayesTraining(X,Y):
    n=len(Y)
    d=len(X[0])
    
    theta_1 = 0
    nb_theta_1 = 0
    theta_1_id = []
    theta_0_id = []
    for i in range(n):
        if Y[i]==1:
            nb_theta_1 += 1
            theta_1_id += [i]
        else:
            theta_0_id += [i]
    theta_1 = nb_theta_1 / n

    theta_i_1_tmp = NumberOfInstance(X,theta_1_id)
    theta_i_0_tmp = NumberOfInstance(X,theta_0_id)
    if (0. in theta_i_1_tmp) or (0. in theta_i_0_tmp):
        print("Warning ! We need Laplace smoothing")
        theta_i_1 = (theta_i_1_tmp+np.full(d,1))/(nb_theta_1+2)
        theta_i_0 = (theta_i_0_tmp+np.full(d,1))/(n-nb_theta_1+2)
    else:
        theta_i_1 = theta_i_1_tmp/nb_theta_1
        theta_i_0 = theta_i_0_tmp/(n-nb_theta_1)    
    
    return (theta_1,theta_i_1,theta_i_0)


###############################################################################
#Says what will happened after having train X and Y and test x_in
###############################################################################              
def NaiveBayesDecision(X,Y,x_in):
    d=len(X[0])

    (theta_1,theta_i_1,theta_i_0) = NaiveBayesTraining(X,Y)
    
    P_1 = theta_1
    P_0 = (1-theta_1)
    for i in range(d):
        if (x_in[i]==1):
            P_1 = P_1*theta_i_1[i]
            P_0 = P_0*theta_i_0[i]
        else:
            P_1 = P_1*(1-theta_i_1[i])
            P_0 = P_0*(1-theta_i_0[i])
    if (P_1>P_0):
        print("succes = 1")
    else:
        print("succes = 0")
 

###############################################################################
#Says what will happened after having train X and Y and test x_in
###############################################################################              
def AddMatrix(observation,prediction):
    MC_tmp = np.zeros((2,2),dtype=int)
    if (prediction==1):
        if (observation==1):
            MC_tmp[0,0] = 1 #True positive
        else:
            MC_tmp[0,1] = 1 #False positive
    else:
        if (observation==1):
            MC_tmp[1,0] = 1 #False negative
        else:
            MC_tmp[1,1] = 1 #True negative
    return MC_tmp
 

###############################################################################
#Compute the confusion matrix of the model
###############################################################################              
def ConfusionMatrix(X_training,Y_training,X_testing,Y_testing):
    d=len(X_training[0])
    p=len(Y_testing)
    Y_prediction = np.zeros(p)
    MC = np.zeros((2,2),dtype=int)

    (theta_1,theta_i_1,theta_i_0) = NaiveBayesTraining(X_training,Y_training)
    
    for k in range(p):
        P_1 = theta_1
        P_0 = (1-theta_1)
        for i in range(d):
            if (X_testing[k,i]==1):
                P_1 = P_1*theta_i_1[i]
                P_0 = P_0*theta_i_0[i]
            else:
                P_1 = P_1*(1-theta_i_1[i])
                P_0 = P_0*(1-theta_i_0[i])
        if (P_1>P_0):
            Y_prediction[k] = 1
        else:
            Y_prediction[k] = 0
        MC += AddMatrix(Y_testing[k],Y_prediction[k])
        
    return MC    
 

###############################################################################
#Compute the accuracy of the model
###############################################################################              
def Accuracy(X_training,Y_training,X_testing,Y_testing,model=0):
    start = time.time()
    
    if model==0:
        MC = ConfusionMatrix(X_training,Y_training,X_testing,Y_testing)
    elif model==1:
        MC = sklearn_models.ConfusionMatrixLogReg(X_training,Y_training,X_testing,Y_testing)
    elif model==2:
        MC = sklearn_models.ConfusionMatrixTree(X_training,Y_training,X_testing,Y_testing)
    
    Acc = (MC[0,0]+MC[1,1])/(MC[0,0]+MC[0,1]+MC[1,0]+MC[1,1])
    Pre = MC[0,0]/(MC[0,0]+MC[0,1])
    Rec = MC[0,0]/(MC[0,0]+MC[1,0])
    F = 2*(Pre*Rec)/(Pre+Rec)
    
    print("Calculus took {0} s".format(round(time.time()-start,4)))
    
    print("Confusion Matrix :\n",MC)

    print('Accuracy = ',Acc)
    print('Precision =',Pre)
    print('Recall = ',Rec)
    print('F1 = ',F)
    
    return F
 

###############################################################################
#Create the file for a Kaggle submission
###############################################################################              
def Submission(X_training,Y_training,X_testing,data_name):
    d=len(X_training[0])
    p=len(Y_training)
    start = time.time()

    (theta_1,theta_i_1,theta_i_0) = NaiveBayesTraining(X_training,Y_training)
    
    with open('submission.csv', 'w') as file:
        file_writer = csv.writer(file, delimiter=',')
        file_writer.writerow(['Id', 'Category'])
        for k in range(p):
            P_1 = theta_1
            P_0 = (1-theta_1)
            for i in range(d):
                if (X_testing[k,i]==1):
                    P_1 = P_1*theta_i_1[i]
                    P_0 = P_0*theta_i_0[i]
                else:
                    P_1 = P_1*(1-theta_i_1[i])
                    P_0 = P_0*(1-theta_i_0[i])
            if (P_1>P_0):
                file_writer.writerow([data_name[k], 1])
            else:
                file_writer.writerow([data_name[k], 0])
    print("Calculus took {0} s".format(round(time.time()-start,4)))
 

###############################################################################
#Function to compute the cross validation
###############################################################################              
def CrossValidation(data_part):
    nb_hf_1_grams = 500
    nb_hf_2_grams = 2250
    nb_hf_3_grams = 4500
    nb_hf_4_grams = 2250
    
    F1 = 0
    step = []
    F1_v = []
        
    max_data = 12500
    nb_folds = 10
    folds = int( data_part * max_data / nb_folds )
    data_range = random.randint(0,max_data-nb_folds*folds)
    
    directory = os.getcwd() + "/Cross_Validation"
    solutionTXT = directory+"/avancement.txt"
    solutionPNG = directory+"/CrossValidation.png"
    if not os.path.exists(directory):
        os.makedirs(directory)
    i = 0
    while (os.path.isfile(solutionTXT))or(os.path.isfile(solutionPNG)):
        solutionTXT = directory+"/avancement_v{0}.txt".format(i)
        solutionPNG = directory+"/CrossValidation_v{0}.png".format(i)
        i+=1
        
    with open(solutionTXT,"w") as file:
        file.write("Cross Validation\n")
        file.write("Number of k-folds = {0}\n".format(nb_folds))
        file.write("\n")

    for kfolds in range(10): #Si on augmente 10 il faut le changer dans ExtractDataTraining
            
            (X_training,Y_training,X_validation,Y_validation) = data_extraction.ExtractDataTraining(data_part,data_range,kfolds,nb_folds,folds,nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams)
            F1_v += [Accuracy(X_training,Y_training,X_validation,Y_validation)]
            F1 += F1_v[-1]
            step += [kfolds]
            with open(solutionTXT,"a") as file:
                file.write("{0} {1}\n".format(step[-1],F1_v[-1]))
                
    F1 = F1/len(F1_v) 
    with open(solutionTXT,"a") as file:
        file.write("----------------> mean = {0}\n".format(F1))
    
    try:
        fig = plt.figure()
        ax = plt.axes()
        ax.set_ylabel('F1')
        ax.set_xlabel('k-folds')
        ax.plot(step,F1_v,label='k - th')
        ax.plot(step,[F1 for k in range(len(step))],label='k - average, -> {0}'.format(round(F1,3)))
        plt.legend()
#        plt.title('{0}% of the dataset'.format(data_part*100))
        plt.savefig(solutionPNG)
        plt.show()
    except:
        return F1
    
    return F1

