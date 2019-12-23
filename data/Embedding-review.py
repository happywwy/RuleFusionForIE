# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from nltk.tokenize import word_tokenize

dic_file = open("Word2Vec_python_code_data/data/w2v_yelp300.txt", "r")
#dic_file = open("Word2Vec_python_code_data/data/w2v_amazon300.txt", "r")
dic = dic_file.readlines()
dic_file.close()

dictionary = {}
count = 0

for line in dic:
    word_vector = line.split(",")[:-1]
    vector_list = []
    for element in word_vector[len(word_vector)-300:]:
        vector_list.append(float(element))
    word = ','.join(word_vector[:len(word_vector)-300])
        
    vector = np.asarray(vector_list)
    dictionary[word] = vector
    

f_train = open("Aspect-Opinion/sentence_res16", "r")
f_test = open("Aspect-Opinion/sentence_restest16", "r")
sentences_train = f_train.read().splitlines()
sentences_test = f_test.read().splitlines()
f_train.close()
f_test.close()

idxs_train = []
idxs_test = []
vocab = ["ppaadd", "punkt", "unk"]
e_pad = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
e_unk = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
e_punkt = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
embedding = [e_pad, e_punkt, e_unk]
for sentence in sentences_train:
    tokens = word_tokenize(sentence.decode('utf-8'))
    idx = []
    for token in tokens:
        token = token.lower()
        if token.isalpha():
            if token not in vocab:
                if token in dictionary.keys():
                    embedding.append(dictionary[token])
                    vocab.append(token)
                    idx.append(vocab.index(token))
                else:
                    count += 1
                    embedding.append(np.asarray([(2 * np.random.rand() - 1) for i in range(300)]))
                    vocab.append(token)
                    idx.append(vocab.index(token))
            else:
                idx.append(vocab.index(token))
            
        else:
            idx.append(vocab.index("punkt"))
    
    idxs_train.append(idx)
    
for sentence in sentences_test:
    tokens = word_tokenize(sentence.decode('utf-8'))
    idx = []
    for token in tokens:
        token = token.lower()
        if token.isalpha():
            if token not in vocab:
                if token in dictionary.keys():
                    embedding.append(dictionary[token])
                    vocab.append(token)
                    idx.append(vocab.index(token))
                else:
                    count += 1
                    idx.append(vocab.index("unk"))
            else:
                idx.append(vocab.index(token))
            
        else:
            idx.append(vocab.index("punkt"))
    
    idxs_test.append(idx)
    
embedding = np.asarray(embedding)

    
print len(vocab)
print count


cPickle.dump(embedding, open("Aspect-Opinion/embedding300_res16", "wb"))
cPickle.dump(idxs_train, open("Aspect-Opinion/idx_res16.train", "wb"))
cPickle.dump(idxs_test, open("Aspect-Opinion/idx_res16.test", "wb"))

