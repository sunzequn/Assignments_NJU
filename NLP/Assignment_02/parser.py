# -*- coding: utf-8 -*-

import nltk
import time
from nltk.tree import Tree
from nltk.metrics import scores
from nltk.parse import viterbi
from os.path import exists
from pickle import dump, load


def generate_grammar(f, s):
    productions = []
    for line in f:
        tree = Tree.fromstring(line)
        # print(tree)
        # 把树chomsky_normal_form
        # tree.collapse_unary(collapsePOS = True, collapseRoot = True)
        # tree.chomsky_normal_form(horzMarkov = 2)
        # tree.set_label('TOP')
        for prodution in tree.productions():
            # if len(prodution.rhs()) == 1 and isinstance(prodution.rhs()[0], nltk.Nonterminal):
            #     print(prodution.rhs())
            productions += [prodution]
    with open('productions.txt', 'w') as f:
        f.write(str(productions))
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
        for part in line.split(' '):
            (w, p) = part.split('_')
            tokens.append(w)
            tags.append(p)
    return tokens, tags


def train():
    t = time.time()
    train_data = open("CTB-auto-pos/allData.txt", "r")
    grammar = generate_grammar(train_data, nltk.Nonterminal('TOP'))
    print("训练花费: %f s" % round(time.time() - t, 2))
    return grammar


if __name__ == '__main__':
    grammar = train()
    test_data = open("CTB-auto-pos/test.txt", "r")
    reference_date = open("CTB-auto-pos/ctb5.1.test.bracketed.txt", "r")
    parser = viterbi.ViterbiParser(grammar)
    precisions = []
    recalls = []
    f_measures = []
    total_time = 0
    for line in test_data:
        tokens, tags = parse_tokens(line)
        t = time.time()
        tree_iterator = parser.parse(tokens)
        tree = next(tree_iterator, False)
        t = round(time.time() - t, 2)
        total_time += t
        if tree:
            parsed = set(constituents(tree))
        else:
            parsed = set()
        ref_tree = Tree.fromstring(reference_date.readline())
        # print(str((tokens, tags, tree, ref)))
        ref_tree = set(constituents(ref_tree))
        precision = scores.precision(ref_tree, parsed)
        precisions.append(precisions)
        recall = scores.recall(ref_tree, parsed)
        recalls.append(recall)
        f_measure = scores.f_measure(ref_tree, parsed)
        f_measures.append(f_measures)
        print(str((''.join(tokens), t, precision, recall, f_measure)))
    print("总耗时: %f s" % round(total_time, 2))

