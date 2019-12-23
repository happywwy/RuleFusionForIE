# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:07:12 2018

@author: wenya
"""
from nltk.tokenize import word_tokenize
import cPickle
import nltk

f_aspect_train = open("Aspect-Opinion/aspectTerm_laptop", "r")
f_aspect_test = open("Aspect-Opinion/aspectTerm_laptest", "r")

f_text_train = open("Aspect-Opinion/sentence_laptop", "r")
f_text_test = open("Aspect-Opinion/sentence_laptest", "r")

sentences_train = f_text_train.read().splitlines()
sentences_test = f_text_test.read().splitlines()
aspects_train = f_aspect_train.read().splitlines()
aspects_test = f_aspect_test.read().splitlines()

f_aspect_train.close()
f_aspect_test.close()
f_text_train.close()
f_text_test.close()

labels_train = []
rels_train = []
rels_test = []
pos_dic = []
pos_train = []
pos_test = []
n = 0
for sentence, aspect in zip(sentences_train, aspects_train):
    tokens = word_tokenize(sentence.decode('utf-8'))
    
    pos = nltk.pos_tag(tokens)
    pos = [item[1] for item in pos]
    pos_train.append(pos)
    
    for item in pos:
        if item not in pos_dic:
            pos_dic.append(item)
    
    label = [0 for i in range(len(tokens))]
    label_type = [0 for i in range(len(tokens))]
    if "NIL" not in aspect:
        aspect_list = aspect.split(",")
        for asp in aspect_list:
            asp = asp.strip()
            if " " not in asp:
                for i, tok in enumerate(tokens):
                    if tok == asp:
                        label[i] = 1
                        label_type[i] = 1
            else:
                asp = asp.split()
                
                for i, tok in enumerate(tokens):
                    if tok == asp[0] and len(tokens) >= i + len(asp):
                        match = True
                        for j in range(1, len(asp)):
                            if tokens[i + j] != asp[j]:
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 1
                        label_type[i] = 1
                        for j in range(1, len(asp)):
                            label[i + j] = 2
                            label_type[i + j] = 1
                            

        
    labels_train.append(label)
    print n
    n += 1
    

labels_test = []
n = 0
for sentence, aspect in zip(sentences_test, aspects_test):
    tokens = word_tokenize(sentence.decode('utf-8'))
    
    pos = nltk.pos_tag(tokens)
    pos = [item[1] for item in pos]
    pos_test.append(pos)
    
    for item in pos:
        if item not in pos_dic:
            pos_dic.append(item)
    
    label = [0 for i in range(len(tokens))]
    label_type = [0 for i in range(len(tokens))]
    if "NIL" not in aspect:
        aspect_list = aspect.split(",")
        for asp in aspect_list:
            asp = asp.strip()
            if " " not in asp:
                for i, tok in enumerate(tokens):
                    if tok == asp:
                        label[i] = 1
                        label_type[i] = 1
            else:
                asp = asp.split()
                for i, tok in enumerate(tokens):
                    if tok == asp[0] and len(tokens) >= i + len(asp):
                        match = True
                        for j in range(1, len(asp)):
                            if tokens[i + j] != asp[j]:
                                match = False
                                break
                    else:
                        match = False
                    if match:
                        label[i] = 1
                        label_type[i] = 1
                        for j in range(1, len(asp)):
                            label[i + j] = 2
                            label_type[i + j] = 1
                            
                    
    labels_test.append(label) 
    print n
    n += 1
    
    
cPickle.dump(labels_train, open("Aspect-Opinion/labels_chunk_laptop.train", "wb"))
cPickle.dump(labels_test, open("Aspect-Opinion/labels_chunk_laptop.test", "wb"))
cPickle.dump((pos_train, pos_test), open("Aspect-Opinion/pos_laptop.tag", "wb"))
