# -*- coding: utf-8 -*-
import cPickle
import os
import numpy as np

f_train = open("ACE05/train.data", "r")
f_dev = open("ACE05/dev.data", "r")
f_test = open("ACE05/test.data", "r")
l_train = f_train.read().splitlines()
l_dev = f_train.read().splitlines()
l_test = f_test.read().splitlines()
f_train.close()
f_dev.close()
f_test.close()

labels_train = []
labels_rel_train = []
words_train = []
pos_train = []

labels_dev = []
labels_rel_dev = []
words_dev = []
pos_dev = []

labels_test = []
labels_rel_test = []
words_test = []
pos_test = []

labels_dic = []
labels_rel_dic = []
pos_dic = []
i = 0
count = 0

while i < len(l_train): 
    text = l_train[i].split()
    pos = l_train[i + 1].split()
    words_train.append(text)
    pos_train.append(pos)
                            
    labels = [0 for l in range(len(text))]
    if l_train[i + 4].rstrip():
        entitys = l_train[i + 4].split('|')
        for entity in entitys:
            entity = entity.split()
            entity_type = entity[-1]
            entity_start = int(entity[0].split(',')[2])
            entity_end = int(entity[0].split(',')[3])
            if entity_type not in labels_dic:
                labels_dic.append(entity_type)
            labels[entity_start] = labels_dic.index(entity_type) * 2 + 1
            if entity_end > entity_start + 1:
                for ind in range(entity_start + 1, entity_end):
                    labels[ind] = labels_dic.index(entity_type) * 2 + 2
                
    labels_train.append(labels)
    
    labels_rel = []
    if l_train[i + 5].rstrip():
        rels = l_train[i + 5].split('|')
        for rel in rels:
            rel = rel.split()
            rel_type = rel[0].split('::')[0]
            rel1 = rel[1].split(',')
            rel1_start, rel1_end = int(rel1[2]), int(rel1[3])
            rel2 = rel[3].split(',')
            rel2_start, rel2_end = int(rel2[2]), int(rel2[3])
            if rel1_start < rel2_start:
                rel_type += '-->'
            else:
                rel_type += '<--'
            if rel_type not in labels_rel_dic:
                labels_rel_dic.append(rel_type)
            ind_rel = labels_rel_dic.index(rel_type) + 1
            if rel1_start < rel2_start:
                labels_rel.append((ind_rel, range(rel1_start, rel1_end), range(rel2_start, rel2_end)))
            else:
                labels_rel.append((ind_rel, range(rel2_start, rel2_end), range(rel1_start, rel1_end)))

    labels_rel_train.append(labels_rel)

    
    i += 7
    print(i)




i = 0
while i < len(l_test): 
    text = l_test[i].split()
    pos = l_test[i + 1].split()
    words_test.append(text)
    pos_test.append(pos)
                            
    labels = [0 for l in range(len(text))]
    if l_test[i + 4].rstrip():
        entitys = l_test[i + 4].split('|')
        for entity in entitys:
            entity = entity.split()
            entity_type = entity[-1]
            entity_start = int(entity[0].split(',')[2])
            entity_end = int(entity[0].split(',')[3])
            if entity_type not in labels_dic:
                labels_dic.append(entity_type)
            labels[entity_start] = labels_dic.index(entity_type) * 2 + 1
            if entity_end > entity_start + 1:
                for ind in range(entity_start + 1, entity_end):
                    labels[ind] = labels_dic.index(entity_type) * 2 + 2
                
    labels_test.append(labels)
    
    labels_rel = []
    if l_test[i + 5].rstrip():
        rels = l_test[i + 5].split('|')
        for rel in rels:
            rel = rel.split()
            rel_type = rel[0].split('::')[0]
            rel1 = rel[1].split(',')
            rel1_start, rel1_end = int(rel1[2]), int(rel1[3])
            rel2 = rel[3].split(',')
            rel2_start, rel2_end = int(rel2[2]), int(rel2[3])
            if rel1_start < rel2_start:
                rel_type += '-->'
            else:
                rel_type += '<--'
            if rel_type not in labels_rel_dic:
                labels_rel_dic.append(rel_type)
            ind_rel = labels_rel_dic.index(rel_type) + 1
            if rel1_start < rel2_start:
                labels_rel.append((ind_rel, range(rel1_start, rel1_end), range(rel2_start, rel2_end)))
            else:
                labels_rel.append((ind_rel, range(rel2_start, rel2_end), range(rel1_start, rel1_end)))
            

    labels_rel_test.append(labels_rel)

    
    i += 7
    print(i)


i = 0
while i < len(l_dev): 
    text = l_test[i].split()
    pos = l_test[i + 1].split()
    words_dev.append(text)
    pos_dev.append(pos)
                            
    labels = [0 for l in range(len(text))]
    if l_dev[i + 4].rstrip():
        entitys = l_dev[i + 4].split('|')
        for entity in entitys:
            entity = entity.split()
            entity_type = entity[-1]
            entity_start = int(entity[0].split(',')[2])
            entity_end = int(entity[0].split(',')[3])
            if entity_type not in labels_dic:
                labels_dic.append(entity_type)
            labels[entity_start] = labels_dic.index(entity_type) * 2 + 1
            if entity_end > entity_start + 1:
                for ind in range(entity_start + 1, entity_end):
                    labels[ind] = labels_dic.index(entity_type) * 2 + 2
                
    labels_dev.append(labels)
    
    labels_rel = []
    if l_dev[i + 5].rstrip():
        rels = l_dev[i + 5].split('|')
        for rel in rels:
            rel = rel.split()
            rel_type = rel[0].split('::')[0]
            rel1 = rel[1].split(',')
            rel1_start, rel1_end = int(rel1[2]), int(rel1[3])
            rel2 = rel[3].split(',')
            rel2_start, rel2_end = int(rel2[2]), int(rel2[3])
            if rel1_start < rel2_start:
                rel_type += '-->'
            else:
                rel_type += '<--'
            if rel_type not in labels_rel_dic:
                labels_rel_dic.append(rel_type)
            ind_rel = labels_rel_dic.index(rel_type) + 1
            if rel1_start < rel2_start:
                labels_rel.append((ind_rel, range(rel1_start, rel1_end), range(rel2_start, rel2_end)))
            else:
                labels_rel.append((ind_rel, range(rel2_start, rel2_end), range(rel1_start, rel1_end)))
            

    labels_rel_dev.append(labels_rel)

    
    i += 7
    print(i)
    
print labels_dic   
print labels_rel_dic

   
cPickle.dump(words_train, open('ACE05/words.train', 'wb'))
cPickle.dump(words_dev, open('ACE05/words.dev', 'wb'))
cPickle.dump(words_test, open('ACE05/words.test', 'wb'))
cPickle.dump(labels_train, open("ACE05/labels_chunk.train", "wb"))
cPickle.dump(labels_dev, open("ACE05/labels_chunk.dev", "wb"))
cPickle.dump(labels_test, open("ACE05/labels_chunk.test", "wb"))

cPickle.dump(labels_rel_train, open('ACE05/labels_2rel.train', 'wb'))
cPickle.dump(labels_rel_dev, open('ACE05/labels_2rel.dev', 'wb'))
cPickle.dump(labels_rel_test, open('ACE05/labels_2rel.test', 'wb'))

cPickle.dump((pos_train, pos_dev, pos_test), open('ACE05/pos.tag', 'wb'))
