# -*- coding: utf-8 -*-


from collections import defaultdict
import numpy as np


def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'L': chunk_end = True
    if prev_tag == 'U': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'U': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'U': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'U': chunk_start = True

    if prev_tag == 'L' and tag == 'L': chunk_start = True
    if prev_tag == 'L' and tag == 'I': chunk_start = True
    if prev_tag == 'U' and tag == 'L': chunk_start = True
    if prev_tag == 'U' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'L': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = float(nb_correct) / nb_pred if nb_pred > 0 else 0
    r = float(nb_correct) / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score
    


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t==y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = float(nb_correct) / nb_true

    return score


def precision_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the precision.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = float(nb_correct) / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the recall.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = float(nb_correct) / nb_true if nb_true > 0 else 0

    return score


def classification_report(y_true, y_pred, digits=4, suffix=False):
    """Build a text report showing the main classification metrics.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.
    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.
    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
        avg / total       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'avg / total'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = float(nb_correct) / nb_pred if nb_pred > 0 else 0
        r = float(nb_correct) / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        if type_name != "Other":
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
            s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report
    
    
 
    

def report_relation_ace(y_true, y_pred, nclass, digits=4):
    
    label_map = {1:'ORG-AFF<--', 2:'PER-SOC-->', 3:'PHYS-->', \
        4:'ORG-AFF-->', 5:'PART-WHOLE-->', 6:'PART-WHOLE<--', \
        7:'ART-->', 8:'ART<--', 9:'GEN-AFF<--', \
        10:'GEN-AFF-->', 11:'PHYS<--', 12:'PER-SOC<--'}
    rel_map = {'ORG-AFF':0, 'PHYS':1, 'GEN-AFF':2, 'PART-WHOLE':3, \
                        'PER-SOC':4, 'ART':5}
    label_names = ['ORG-AFF', 'PHYS', 'GEN-AFF', 'PART-WHOLE', 'PER-SOC', 'ART']
    
    relevant = np.asarray([0] * (nclass - 1))
    correct = np.asarray([0] * (nclass - 1))
    predict = np.asarray([0] * (nclass - 1))
    
    for true_i, pred_i in zip(y_true, y_pred):
        #true_i = set(true_i)
        #pred_i = set(pred_i)

        for tup_true in true_i:
            if tup_true[0] != 0:
                cat = label_map[tup_true[0]][:-3]
                relevant[rel_map[cat]] += 1
        for key in pred_i.keys():
            if pred_i[key] != 0:
                cat = label_map[pred_i[key]][:-3]
                predict[rel_map[cat]] += 1
        for key in pred_i.keys():
            (label, s, e, b_idx) = key
            if pred_i[key] != 0:
                cat = label_map[pred_i[key]][:-3]
                if label_map[pred_i[key]][-3:] == '-->':
                    rlabel = cat + '<--'
                else:
                    rlabel = cat + '-->'
                for key2 in label_map.keys():
                    if label_map[key2] == rlabel:
                        r = key2
                        break
                if (pred_i[key], list(s), list(e)) in true_i or (r, list(s), list(e)) in true_i:
                    correct[rel_map[cat]] += 1
            
    p = float(correct.sum()) / predict.sum() if predict.sum() > 0 else 0
    r = float(correct.sum()) / relevant.sum() if relevant.sum() > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    
    last_line_heading = 'avg / total'
    name_width = max([len(key) for key in rel_map.keys()])
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for i in range(nclass - 1):
        type_name = label_names[i]
        pi = float(correct[i]) / predict[i] if predict[i] > 0 else 0
        ri = float(correct[i]) / relevant[i] if relevant[i] > 0 else 0
        f1i = 2 * pi * ri / (pi + ri) if pi + ri > 0 else 0

        report += row_fmt.format(*[type_name, pi, ri, f1i, relevant[i]], width=width, digits=digits)

        ps.append(pi)
        rs.append(ri)
        f1s.append(f1i)
        s.append(relevant[i])

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)


    return p, r, f1, report

def report_relation_trec(y_true, y_pred, nclass, digits=4):
    
    label_map = {1:'Live_In<--', 2:'OrgBased_In-->', \
        3:'Located_In-->', 4:'OrgBased_In<--', 5:'Located_In<--', \
        6:'Live_In-->', 7:'Work_For-->', 8:'Work_For<--', 9:'Kill-->', 10:'Kill<--'}
    
    rel_map = {'Live_In':0, 'OrgBased_In':1, 'Located_In':2, 'Work_For':3, \
                        'Kill':4}
    label_names = ['Live_In', 'OrgBased_In', 'Located_In', 'Work_For', 'Kill']
    
    relevant = np.asarray([0] * (nclass - 1))
    correct = np.asarray([0] * (nclass - 1))
    predict = np.asarray([0] * (nclass - 1))
    
    for true_i, pred_i in zip(y_true, y_pred):
        for tup_true in true_i:
            (label, s, e) = tup_true
            if tup_true[0] != 0:
                cat = label_map[tup_true[0]][:-3]
                relevant[rel_map[cat]] += 1
        for key in pred_i.keys():
            if pred_i[key] != 0:
                cat = label_map[pred_i[key]][:-3]
                predict[rel_map[cat]] += 1
        for key in pred_i.keys():
            (label, s, e, b_idx) = key
            if pred_i[key] != 0:
                cat = label_map[pred_i[key]][:-3]
                
                if label_map[pred_i[key]][-3:] == '-->':
                    rlabel = cat + '<--'
                else:
                    rlabel = cat + '-->'
                for key2 in label_map.keys():
                    if label_map[key2] == rlabel:
                        r = key2
                        break
                
                if (pred_i[key], list(s), list(e)) in true_i or (r, list(s), list(e)) in true_i:
                    correct[rel_map[cat]] += 1
                        
    p = float(correct.sum()) / predict.sum() if predict.sum() > 0 else 0
    r = float(correct.sum()) / relevant.sum() if relevant.sum() > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    
    last_line_heading = 'avg / total'
    name_width = max([len(key) for key in rel_map.keys()])
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for i in range(nclass - 1):
        type_name = label_names[i]
        pi = float(correct[i]) / predict[i] if predict[i] > 0 else 0
        ri = float(correct[i]) / relevant[i] if relevant[i] > 0 else 0
        f1i = 2 * pi * ri / (pi + ri) if pi + ri > 0 else 0

        report += row_fmt.format(*[type_name, pi, ri, f1i, relevant[i]], width=width, digits=digits)

        ps.append(pi)
        rs.append(ri)
        f1s.append(f1i)
        s.append(relevant[i])

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)


    return p, r, f1, report
    
