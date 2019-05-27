# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:14:41 2019

@author: Alexandre
"""

import numpy as np
import os
import time
import matrix_creation


###############################################################################
#Get the "OpinionLexicon" of Hiu
###############################################################################
def GetOpinionLexicon():
    directory = os.getcwd()+"/OpinionLexicon"
    with open(directory+"/"+"positive-words.txt","r") as file:
        while True:
            line = file.readline()
            if not line.startswith(";"):
                break
        positive = file.read().split("\n")
    with open(directory+"/"+"negative-words.txt","r") as file:
        while True:
            line = file.readline()
            if not line.startswith(";"):
                break
        negative = file.read().split("\n")
    return (positive,negative)


###############################################################################
#Get the data from the training set
###############################################################################
def ExtractDataTraining(data_part,data_range,kfolds,nb_folds,folds,nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams,test=False):
    Y_training = []
    Y_testing = []
    data_training = []
    data_testing = []
    start = time.time()
    
    k=0
    directory = os.getcwd()+"/train/train/pos"
    for name in os.listdir(directory)[data_range:data_range+nb_folds*folds]:
        with open(directory+"/"+name,"r",encoding='utf-8') as file:
            if (kfolds * folds <= k) and (k < (kfolds + 1) * folds):
                try:
                    data_testing += [file.read()]
                    Y_testing += [1]
                except:
                    print("pos/"+name+" is unreadable")
                    data_testing += ["good"]
                    Y_testing += [1]
            else:
                try:
                    data_training += [file.read()]
                    Y_training += [1]
                except:
                    print("pos/"+name+" is unreadable")
                    data_training += ["good"]
                    Y_training += [1]
            k+=1
               
    k=0
    directory = os.getcwd()+"/train/train/neg"
    for name in os.listdir(directory)[data_range:data_range+nb_folds*folds]:
        with open(directory+"/"+name,"r",encoding='utf-8') as file:
            if (kfolds * folds <= k) and (k < (kfolds + 1) * folds):
                try:
                    data_testing += [file.read()]
                    Y_testing += [0]
                except:
                    print("neg/"+name+" is unreadable")
                    data_testing += ["bad"]
                    Y_testing += [0]
            else:
                try:
                    data_training += [file.read()]
                    Y_training += [0]
                except:
                    print("neg/"+name+" is unreadable")
                    data_training += ["bad"]
                    Y_training += [0]
            k+=1  
                    
    print("Extraction took {0} s".format(round(time.time()-start,4)))
    if test==True:
        return (data_training,Y_training,data_testing,Y_testing)

    else:
        start = time.time()
        Y_training = np.array(Y_training)
        Y_testing = np.array(Y_testing)
        (X_training,X_testing) = matrix_creation.CreateX(data_training,data_testing,nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams)
        print("Creation of all X and Y took {0} s".format(round(time.time()-start,4)))
        return (X_training,Y_training,X_testing,Y_testing)


###############################################################################
#Get the data from the training set and the testing set
###############################################################################
def ExtractDataTesting(nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams):
    Y_training = []
    data_training = []
    data_testing = []
    data_name = []
    start = time.time()

    directory = os.getcwd()+"/train/train/pos" # pos1 contient les 4 premiers fichiers de pos
    for name in os.listdir(directory):
        with open(directory+"/"+name,"r",encoding='utf-8') as file:
            try:
                data_training += [file.read()]
                Y_training += [1]
            except:
                print("pos/"+name+" is unreadable")
                data_training += ["good"]
                Y_training += [1]
    directory = os.getcwd()+"/train/train/neg" # neg1 contient les 4 premiers fichiers de neg
    for name in os.listdir(directory):
        with open(directory+"/"+name,"r",encoding='utf-8') as file:
            try:
                data_training += [file.read()]
                Y_training += [0]    
            except:
                print("neg/"+name+" is unreadable")
                data_training += ["bad"]
                Y_training += [0]    
                
    directory = os.getcwd()+"/test/test"
    for name in os.listdir(directory): # Les 25000 données sont la "trainingdata"
        with open(directory+"/"+name,"r",encoding='utf-8') as file:
            try:
                data_testing += [file.read()]
                data_name += [name.replace(".txt", "")]
            except:
                print(name+" is unreadable")
                data_testing += [" "]

    print("Extraction took {0} s".format(round(time.time()-start,4)))

    start = time.time()
    Y_training = np.array(Y_training)
    (X_training,X_testing) = matrix_creation.CreateX(data_training,data_testing,nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams)
    print("Creation of X and Y took {0} s".format(round(time.time()-start,4)))
    return (X_training,Y_training,X_testing,data_name)
