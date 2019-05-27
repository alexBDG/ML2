# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:25:17 2019

@author: Alexandre
"""

import numpy as np
import sys
from nltk.tokenize import word_tokenize
import time
import data_preprocessing
import data_extraction


###############################################################################
#Create the matrix of the features
###############################################################################              
def CreateX(data,data_validation,nb_hf_1_grams,nb_hf_2_grams,nb_hf_3_grams,nb_hf_4_grams):
    n = len(data)
    data_cleaned = data_preprocessing.CleanWords(data)
    data_validation_cleaned = data_preprocessing.CleanWords(data_validation)
    print("words cleaned")
    data_lemmatized = data_preprocessing.StemmerWords(data_cleaned)
    print("words lemmatized")

    #feature 1
    (positive,negative) = data_extraction.GetOpinionLexicon()
    
    #feature 2->nb_w_max+nb_w_min+1
    hf_positive = data_preprocessing.find_frequent_pos_neg(data_lemmatized,nb_hf_1_grams,positive)
    lf_negative = data_preprocessing.find_frequent_pos_neg(data_lemmatized,nb_hf_1_grams,negative)
    print("most frequent words found")
    
    #feature Bigrams
    (k2_grams,hf_2_grams,k2_grams_validation) = data_preprocessing.KGrams(2,nb_hf_2_grams,data,data_validation)
    nb_hf_2_grams = len(hf_2_grams)
    print("2-grams found")
    
    #feature Trigrams
    (k3_grams,hf_3_grams,k3_grams_validation) = data_preprocessing.KGrams(3,nb_hf_3_grams,data,data_validation)
    nb_hf_3_grams = len(hf_3_grams)
    print("3-grams found")

    #feature Quadrigrams
    (k4_grams,hf_4_grams,k4_grams_validation) = data_preprocessing.KGrams(4,nb_hf_4_grams,data,data_validation)
    nb_hf_4_grams = len(hf_4_grams)
    print("4-grams found")
    
    X = np.zeros((n,2+2*nb_hf_1_grams+nb_hf_2_grams+nb_hf_3_grams+nb_hf_4_grams),dtype=int)
    
###############################################################################

    
    print("Creation of X - training")
    start = time.time()
    for k in range(n):
        ph = "\rProgression: {0} % ".format(round(float(100*k)/float(n-1),3))
        sys.stdout.write(ph)
        sys.stdout.flush()
        
        #feature 1
        positive_words = len(set(data_cleaned[k]) & set(positive))
        negative_words = len(set(data_cleaned[k]) & set(negative))
        if (positive_words>negative_words):
            X[k,0] = 1
        else:
            X[k,0] = 0

        #feature 2->nb_w_max+nb_w_min+1
        i=0
        for word in [w.lower() for w in word_tokenize(data[k]) if w not in ['.',',',';',"'",'"','?','!','/','<','>',':','\\']]:
            i=0
            if (word == "spoiler"):
                X[k,1] = 1
            for hfw in hf_positive:
                if (hfw == word):
                    X[k,2+i] = 1
                i+=1
            i=0
            for lfw in lf_negative:
                if (lfw == word):
                    X[k,i+2+nb_hf_1_grams] = 1
                i+=1
                
        #feature Bigrams
        i=0
        for gram in k2_grams[k]:
            i=0
            for hfw in hf_2_grams:
                if (hfw == gram):
                    X[k,i+2+2*nb_hf_1_grams] = 1
                i+=1
                
        #feature Trigrams
        i=0
        for gram in k3_grams[k]:
            i=0
            for hfw in hf_3_grams:
                if (hfw == gram):
                    X[k,i+2+2*nb_hf_1_grams+nb_hf_2_grams] = 1
                i+=1
                
        #feature Quadrigrams
        i=0
        for gram in k4_grams[k]:
            i=0
            for hfw in hf_4_grams:
                if (hfw == gram):
                    X[k,i+2+2*nb_hf_1_grams+nb_hf_2_grams+nb_hf_3_grams] = 1
                i+=1

    print("----> took {0} s".format(round(time.time()-start,4)))

###############################################################################
    n = len(data_validation)
    
    X_validation = np.zeros((n,2+2*nb_hf_1_grams+nb_hf_2_grams+nb_hf_3_grams+nb_hf_4_grams),dtype=int)
    
    print("Creation of X - validation")
    start = time.time()
    
    for k in range(n):
        ph = "\rProgression: {0} % ".format(round(float(100*k)/float(n-1),3))
        sys.stdout.write(ph)
        sys.stdout.flush()

        #feature 1 - VALIDATION
        positive_words = len(set(data_validation_cleaned[k]) & set(positive))
        negative_words = len(set(data_validation_cleaned[k]) & set(negative))
        if (positive_words>negative_words):
            X_validation[k,0] = 1
        else:
            X_validation[k,0] = 0
            
        #feature 2->nb_w_max+nb_w_min+1
        i=0
        for word in [w.lower() for w in word_tokenize(data_validation[k]) if w not in ['.',',',';',"'",'"','?','!','/','<','>',':','\\']]:
            i=0
            if (word == "spoiler"):
                X_validation[k,1] = 1
            for hfw in hf_positive:
                if (hfw == word):
                    X_validation[k,2+i] = 1
                i+=1
            i=0
            for lfw in lf_negative:
                if (lfw == word):
                    X_validation[k,i+2+nb_hf_1_grams] = 1
                i+=1
        
        #feature Bigrams
        i=0
        for gram in k2_grams_validation[k]:
            i=0
            for hfw in hf_2_grams:
                if (hfw == gram):
                    X_validation[k,i+2+2*nb_hf_1_grams] = 1
                i+=1
                
        #feature Trigrams
        i=0
        for gram in k3_grams_validation[k]:
            i=0
            for hfw in hf_3_grams:
                if (hfw == gram):
                    X_validation[k,i+2+2*nb_hf_1_grams+nb_hf_2_grams] = 1
                i+=1
                
        #feature Quadrigrams
        i=0
        for gram in k4_grams_validation[k]:
            i=0
            for hfw in hf_4_grams:
                if (hfw == gram):
                    X_validation[k,i+2+2*nb_hf_1_grams+nb_hf_2_grams+nb_hf_3_grams] = 1
                i+=1

    print("----> took {0} s".format(round(time.time()-start,4)))
                
###############################################################################

    return (X,X_validation)

