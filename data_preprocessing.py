# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:15:46 2019

@author: Alexandre
"""

import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
from collections import Counter


###############################################################################
#Remove the stopwords from the data
###############################################################################
def CleanWords(data):
    new_data =[]
    stopw = stopwords.words('english')
    for text in data:
#        text = text.split(' ')
        words = word_tokenize(text)
        tokens = [w.lower() for w in words if w.isalpha()]
        new_data += [[token for token in tokens if token not in stopw]]
    return new_data


###############################################################################
#stemmerize the data
###############################################################################
def StemmerWords(data):
    porter_stemmer = PorterStemmer()
    new_data = []
    for text in data:
        new_data += [[porter_stemmer.stem(w) for w in text]]
    return new_data


###############################################################################
#Find the list of the "nb_w_max" most frequent words in "data" without in the
#the list "pos_neg"
###############################################################################
def find_frequent_pos_neg(data,nb_w_max,pos_neg):
    words = {} 
    hf_words = {} 
    for text in data:
#        tokens = word_tokenize(text)
        tokens = [w.lower() for w in text]
        for word in [word for word in tokens if word in pos_neg]:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
                            
    hf_words = Counter(words).most_common(nb_w_max)
    hf_words = [hf[0] for hf in hf_words]
        
    return(hf_words)
    
    
###############################################################################
#Find the most frequent ngrams that are not in the same time in the list from
#the positiv review and from the negativ review            
###############################################################################   
def KGrams(k,nb_hf_k_grams,data,data_validation):

    (k_grams_pos,hf_k_grams_pos) = SplitInKgrams(k,nb_hf_k_grams,data[:int(len(data)/2)],True)
    (k_grams_neg,hf_k_grams_neg) = SplitInKgrams(k,nb_hf_k_grams,data[int(len(data)/2):],True)
    k_grams = k_grams_pos + k_grams_neg
    hf_k_grams_pos = [w for w in hf_k_grams_pos]
    hf_k_grams_neg = [w for w in hf_k_grams_neg]
    hf_k_grams = (set(hf_k_grams_pos) - set(hf_k_grams_neg)) | (set(hf_k_grams_neg) - set(hf_k_grams_pos))
    k_grams_validation = SplitInKgrams(k,nb_hf_k_grams,data_validation,False)
    
    return (k_grams,hf_k_grams,k_grams_validation)
    
    
def SplitInKgrams(k,nb_hf_k_grams,data,IsTraining):
    stop = ['...','--','``',"''",'br','.',',',';',"'",'"','?','!','/','<','>',':','\\','(',')','[',']']
    k_grams = []
    grams = {}
    hf_k_grams = {}
    
    print("Split in {0}-grams".format(k))
    start = time.time()
    p=0
    for text in data:
        ph = "\rProgression: {0} % ".format(round(float(100*p)/float(len(data)-1),3))
        sys.stdout.write(ph)
        sys.stdout.flush()

        tokens = word_tokenize(text.lower())
        k_gram_sentence = []
        i=k-1
        while (i<len(tokens)):
            if (k==1):
                if (tokens[i] not in stop):
                    k_gram = tokens[i]
                    k_gram_sentence += [k_gram]
                    if k_gram in grams:
                        grams[k_gram] += 1
                    else:
                        grams[k_gram] = 1

            if (k==2):
                if (tokens[i] not in stop)and(tokens[i-1] not in stop):
                    k_gram = tokens[i-1] + "_" + tokens[i]
                    k_gram_sentence += [k_gram]
                    if k_gram in grams:
                        grams[k_gram] += 1
                    else:
                        grams[k_gram] = 1
                        
            if (k==3):
                if (tokens[i] not in stop)and(tokens[i-1] not in stop)and(tokens[i-2] not in stop):
                    k_gram = tokens[i-2] + "_" + tokens[i-1] + "_" + tokens[i]
                    k_gram_sentence += [k_gram]
                    if k_gram in grams:
                        grams[k_gram] += 1
                    else:
                        grams[k_gram] = 1                        
            if (k==4):
                if (tokens[i] not in stop)and(tokens[i-1] not in stop)and(tokens[i-2] not in stop)and(tokens[i-3] not in stop):
                    k_gram = tokens[i-3] + "_" + tokens[i-2] + "_" + tokens[i-1] + "_" + tokens[i]
                    k_gram_sentence += [k_gram]
                    if k_gram in grams:
                        grams[k_gram] += 1
                    else:
                        grams[k_gram] = 1
                        
            if (k==5):
                if (tokens[i] not in stop)and(tokens[i-1] not in stop)and(tokens[i-2] not in stop)and(tokens[i-3] not in stop)and(tokens[i-4] not in stop):
                    k_gram = tokens[i-4] + "_" + tokens[i-3] + "_" + tokens[i-2] + "_" + tokens[i-1] + "_" + tokens[i]
                    k_gram_sentence += [k_gram]
                    if k_gram in grams:
                        grams[k_gram] += 1
                    else:
                        grams[k_gram] = 1
        
            i+=1
        k_grams += [k_gram_sentence]
        p+=1
    print("----> took {0} s".format(round(time.time()-start,4)))
        
    if (IsTraining==1):
        print("Find the {0} most frequent {1}-grams".format(nb_hf_k_grams,k))
        start = time.time()
        hf_k_grams = Counter(grams).most_common(nb_hf_k_grams)
        hf_k_grams = [hf[0] for hf in hf_k_grams]
        print("Progression: 100.0 %  ----> took {0} s".format(round(time.time()-start,4)))
    else:
        return k_grams

    return (k_grams,hf_k_grams)

