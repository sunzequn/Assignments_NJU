#!/usr/bin/python3
import nltk
import time
from nltk.tree import Tree
from nltk.metrics import scores
from nltk.parse import viterbi


def constituents(tree, start=0):
    yield (tree.label(), start, start + len(tree.leaves()))
    for leave in tree:
        if isinstance(leave, str):
            break
        for constituent in constituents(leave, start):
            yield constituent
        start += len(leave.leaves())

def construct_grammar(f, S):
    productions = []
    for line in f:
        tree = Tree.fromstring(line)
        # 把树chomsky_normal_form
        # tree.collapse_unary(collapsePOS = True, collapseRoot = True)
        # tree.chomsky_normal_form(horzMarkov = 2)
        # tree.set_label('TOP')
        for prodution in tree.productions():
            if len(prodution.rhs()) == 1 and isinstance(prodution.rhs()[0], nltk.Nonterminal):
                print(prodution.rhs())
            productions += [prodution]
    with open('productions.txt', 'w') as f:
        f.write(str(productions))
    return nltk.induce_pcfg(S, productions)

def get_tokens(line):
    tokens = []
    tags = []
    if line.endswith('\n'):
        line = line[:len(line) - 1]
        for part in line.split(' '):
            (w, p) = part.split('_')
            tokens.append(w)
            tags.append(p)
    return (tokens, tags)

beam_size = 60

from os.path import exists
from pickle import dump, load

grammar = None
if exists('grammar.bin'):
    try:
        with open('grammar.bin', 'rb') as f:
            grammar = load(f)
            pass
    except EOFError:
        grammar = None

if grammar == None:
    with open("CTB-auto-pos/allData.txt", "r") as f:
        grammar = construct_grammar(f, nltk.Nonterminal('TOP'))
        with open('grammar.bin', 'wb') as o:
            dump(grammar, o)
            pass

with open("productions.txt", "w") as o:
    for production in grammar.productions():
        o.write(str(production) + '\n')

print('grammar loaded')

print(grammar.is_chomsky_normal_form())


#parser = pchart.InsideChartParser(grammar, trace=5, beam_size=40)
#parser = chart.BottomUpChartParser(grammar)
parser = viterbi.ViterbiParser(grammar)

with open("CTB-auto-pos/test.txt", "r") as test, \
    open("CTB-auto-pos/ctb5.1.test.bracketed.txt", "r") as reference, \
    open("accuracy-%d.txt" % beam_size, "w") as accuracy, \
        open("output-%d.txt" % beam_size, "w") as output:
    for line in test:
        tokens = []
        tags = []
        if line.endswith('\n'):
            line = line[:len(line) - 1]
        for part in line.split(' '):
            (w, p) = part.split('_')
            tokens.append(w)
            tags.append(p)
        print(''.join(tokens))
        time_begin = time.time()
        tree_iterator = parser.parse(tokens)
        tree = next(tree_iterator, False)
        time_cost = time.time() - time_begin
        if tree:
            parsed = set(constituents(tree))
        else:
            parsed = set()
        ref = Tree.fromstring(reference.readline())
        output.write(str((tokens, tags, tree, ref)))
        output.write('\n')
        output.flush()
        ref = set(constituents(ref))
        m = len(parsed)
        n = len(ref)
        precision = scores.precision(ref, parsed)
        recall = scores.recall(ref, parsed)
        f_measure = scores.f_measure(ref, parsed)
        accuracy.write(str((''.join(tokens), time_cost, m, n, precision, recall, f_measure)))
        accuracy.write('\n')
        accuracy.flush()
