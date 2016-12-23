# -*- coding: utf-8 -*-

import nltk
import time
from nltk.tree import Tree
from nltk.metrics import scores
from nltk.parse import viterbi


def generate_grammar(f, s):
    productions = []
    for line in f:
        tree = Tree.fromstring(line)
        for prodution in tree.productions():
            productions += [prodution]
    return nltk.induce_pcfg(s, productions)


def constituents(tree, start=0):
    yield (tree.label(), start, start + len(tree.leaves()))
    for leave in tree:
        if isinstance(leave, str):
            break
        for constituent in constituents(leave, start):
            yield constituent
        start += len(leave.leaves())


def parse_tokens(line):
    tokens = []
    tags = []
    if line.endswith('\n'):
        line = line[:len(line) - 1]
        for param in line.split(' '):
            token, tag = param.split('_')
            tokens.append(token)
            tags.append(tag)
    return tokens, tags


def train():
    t = time.time()
    train_data = open("CTB-auto-pos/allData.txt", "r", encoding='utf8')
    grammar = generate_grammar(train_data, nltk.Nonterminal('TOP'))
    print("训练花费: %f s" % round(time.time() - t, 2))
    return grammar


def avg(ll):
    num = 0
    for l in ll:
        num += l
    return round(num / len(ll), 4)


if __name__ == '__main__':
    grammar = train()
    test_data = open("CTB-auto-pos/test.txt", "r", encoding='utf8')
    reference_date = open("CTB-auto-pos/ctb5.1.test.bracketed.txt", "r", encoding='utf8')
    parser = viterbi.ViterbiParser(grammar)
    precisions = []
    recalls = []
    f_measures = []
    tokens_len = []
    times = []
    total_time = 0
    for line in test_data:
        tokens, tags = parse_tokens(line)
        if len(tokens) > 15:
            reference_date.readline()
            continue
        tokens_len.append(len(tokens))
        t = time.time()
        tree_iterator = parser.parse(tokens)
        tree = next(tree_iterator, False)
        t = round(time.time() - t, 2)
        total_time += t
        times.append(t)
        if tree:
            parsed = set(constituents(tree))
        else:
            parsed = set()
        reference = Tree.fromstring(reference_date.readline())
        reference = set(constituents(reference))
        precision = round(scores.precision(reference, parsed), 4)
        precisions.append(precision)
        recall = round(scores.recall(reference, parsed), 4)
        recalls.append(recall)
        f_measure = round(scores.f_measure(reference, parsed), 4)
        f_measures.append(f_measure)
        print(str((''.join(tokens), t, precision, recall, f_measure)))
    print("测试数据条数: %d" % len(precisions))
    print("总耗时: %f s" % round(total_time, 2))
    print("平均token长度: %f" % avg(tokens_len))
    print("平均耗时: %f s" % avg(times))
    print("平均精度: %f" % avg(precisions))
    print("平均召回率: %f" % avg(recalls))
    print("平均f-measure: %f" % avg(f_measures))
