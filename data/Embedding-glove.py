# -*- coding: utf-8 -*-

import numpy as np
import cPickle


dic_file = open("glove.840B.300d.txt", "r")
dic = dic_file.readlines()
dic_file.close()

dictionary = {}
count = 0

for line in dic:
    word_vector = line.split(" ")
    vector_list = []
    for element in word_vector[len(word_vector)-300:]:
        vector_list.append(float(element))
    word = ','.join(word_vector[:len(word_vector)-300])
        
    vector = np.asarray(vector_list)
    dictionary[word] = vector
    

#words_train = cPickle.load(open("TREC/words.train", "rb"))
#words_test = cPickle.load(open("TREC/words.test", "rb"))
words_train = cPickle.load(open("ACE05/words.train", "rb"))
words_test = cPickle.load(open("ACE05/words.test", "rb"))


idxs_train = []
idxs_test = []
vocab = ["ppaadd", "punkt", "unk"]
e_pad = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
e_unk = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
e_punkt = np.asarray([(2 * np.random.rand() - 1) for i in range(300)])
embedding = [e_pad, e_punkt, e_unk]

for tokens in words_train:
    idx = []
    for token in tokens:
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
    
for tokens in words_test:
    idx = []
    for token in tokens:
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

    
print(len(vocab))
print(count)


cPickle.dump(embedding, open("ACE05/embedding300_glove", "wb"))
cPickle.dump(idxs_train, open("ACE05/idx.train", "wb"))
cPickle.dump(idxs_test, open("ACE05/idx.test", "wb"))
print(len(idxs_train))
print(len(idxs_test))

