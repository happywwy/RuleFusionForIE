# -*- coding: utf-8 -*-
"""
Credit to "Extracting Entities and Relations with Joint Minimum Risk Training"
"""
from collections import defaultdict
import re
import numpy as np

def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    '''
    if prev_tag == 'O' and tag == 'I': chunk_start = True
    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True
    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True
    '''
    return chunk_start

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False
    
    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True
    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True
    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end
    
def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')
    
    
def schedule_sample(sch_k, pred_tags, Y, i_epoch):
    sch_p = sch_k / (sch_k + np.exp(i_epoch / sch_k))
    sch_tags = []
    for i, tags in enumerate(pred_tags):
        each_tags = []
        for j, tag in enumerate(tags):
            rd = np.random.random()
            if rd <= sch_p:
                each_tags.append(Y[i][j])
            else:
                each_tags.append(tag)
        sch_tags.append(each_tags)
    return sch_tags
    
def generate_candidate_entity_pair(Y, Y_rel, label_map, entity_map, training=False):
    labels = []
    idx2batch = {}
    entity_pair_idxs = []
    for batch_idx in range(len(Y)):
        rel_dict = generate_relation_dict(Y_rel[batch_idx])
        y = [label_map[t.item()] for t in Y[batch_idx]]
        t_entity = get_entity(y)

        entity_idx2chunk_type = get_entity_idx2chunk_type(t_entity, entity_map)
        instance_candidate_set = generate_candidate_entity_pair_with_win(entity_idx2chunk_type)

        if training:
            add_gold_candidate(instance_candidate_set, Y_rel[batch_idx])

        for b, e in instance_candidate_set:
            if (b, e) in rel_dict:
                t = rel_dict[(b, e)]
            else:
                t = 0
            idx2batch[len(entity_pair_idxs)] = batch_idx
            entity_pair_idxs.append((b, e))
            labels.append(t)
    return entity_pair_idxs, labels, idx2batch
    
def get_entity(y):
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus
    guessed_idx = []
    t_guessed_entity2idx = defaultdict(list)
    for i, tag in enumerate(y):
        guessed, guessed_type = parse_tag(tag)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                            last_guessed_type, guessed_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                        last_guessed_type, guessed_type)
        if start_guessed:
            if guessed_idx:
                t_guessed_entity2idx[guessed_idx[0]].append(tuple(guessed_idx[1:]))
            guessed_idx = [guessed_type, i]
        elif guessed_idx and not start_guessed and guessed_type == guessed_idx[0]:
            guessed_idx.append(i)

        last_guessed = guessed
        last_guessed_type = guessed_type
    if guessed_idx:
        t_guessed_entity2idx[guessed_idx[0]].append(tuple(guessed_idx[1:]))
    return t_guessed_entity2idx
    
def add_gold_candidate(instance_candidate_set, y_rel):
    for t, b, e in y_rel:
        instance_candidate_set.add((tuple(b), tuple(e)))

def generate_relation_dict(y_rel):
    rel_dict = {}
    for t, b, e in y_rel:
        rel_dict[(tuple(b), tuple(e))] = t
    return rel_dict
    
def get_entity_idx2chunk_type(t_entity, label_map):
    entity_idx2chunk_type = {}
    for k, v in t_entity.items():
        for e in v:
            entity_idx2chunk_type[e] = label_map[k]
    return entity_idx2chunk_type
    
def generate_candidate_entity_pair_with_win(entity_idx2chunk_type):
    instance_candidate_set = set()
    for ent1_idx in entity_idx2chunk_type.keys():
        for ent2_idx in entity_idx2chunk_type.keys():
            if ent1_idx[0] >= ent2_idx[0]:
                continue
            instance_candidate_set.add((ent1_idx, ent2_idx))
    return instance_candidate_set
