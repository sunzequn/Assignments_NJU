# encoding=utf-8
import os
import re
import math
import time
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# the folder of papers
folder_prefix = "./ICML"
# output file
out_file = 'results.txt'
# the dist of categories, whose keys are numbers, e.g., 1,2,3,...,15
categories = {}
# the key of the category to output
category_out = '7'
# the number of all papers
total_papers_num = 0
# the set of words appear in all papers
all_words = set()
# IDF values of all words
words_idf = {}
# the list of words in the lexicographical order
words_index = []


# calculate IDF values of all words
def idf(word, categories):
    num = 0
    for key in categories:
        for paper in categories[key].papers:
            if word in paper.words_tf:
                num += 1
    words_idf[word] = math.log(total_papers_num / num)


# calculate TF values of the word list
def tf(words):
    words_tf = {}
    # the number of words in the paper
    num = len(words)
    for word in words:
        # add the word to global words list
        all_words.add(word)
        if word in words_tf:
            words_tf[word] += 1 / num
        else:
            words_tf[word] = 1 / num
    return words_tf


# calculate TF-IDF values of words of the paper
def tf_idf(paper):
    words_tf_idf = {}
    for word in paper.words_tf:
        words_tf_idf[word] = paper.words_tf[word] * words_idf[word]
    return words_tf_idf


# text pre-processing
def handle_words(content):
    # word segmentation
    tokens = nltk.word_tokenize(content)
    words = []
    stemmer = SnowballStemmer('english')
    for token in tokens:
        # remove stopwords and strings containing non-alphanumeric characters, or whose length is 1
        if re.search(u'^[a-zA-Z]+$', token) and len(token) > 1 and token.lower() not in stopwords.words('english'):
            # translate words into lower case and extract stemmer
            words.append(stemmer.stem(token.lower()))
    return words


def out_category(category_key):
    # in the lexicographical order
    papers_out = sorted(categories[category_key].papers, key=lambda x: x.name)
    for paper in papers_out:
        print("output paper : " + paper.name)
        out_paper(tf_idf(paper))


def out_paper(words_tf_idf):
    f = open(out_file, 'a')
    f.write('[')
    line = ''
    words_index_tf_idf = {}
    for word in words_tf_idf:
        words_index_tf_idf[words_index.index(word) + 1] = words_tf_idf[word]
    words_index_ordered_tf_idf = sorted(words_index_tf_idf.items(), key=lambda w: w[0])
    num = 0
    for words_index_ordered in words_index_ordered_tf_idf:
        if words_index_ordered[1] > 0:
            if num > 0:
                line += ', '
            line += "%d:%.10f" % (words_index_ordered[0], words_index_ordered[1])
            num += 1
    f.write(line)
    f.write(']\n')
    f.close()


def out_stopwords():
    f = open('stopwords.txt', 'a')
    for stopword in stopwords.words('english'):
        f.write(stopword + '\n')
    f.close()


def out_words_index():
    f = open('words_index.txt', 'a')
    num = 1
    for word in words_index:
        f.write(str(num) + '\t' + word + '\n')
        num += 1
    f.close()


class Paper:
    def __init__(self, file, path_prefix):
        global total_papers_num
        total_papers_num += 1
        self.name = file.replace('.txt', '')
        self.words_tf = tf(handle_words(open(os.path.join(path_prefix, file)).read()))
        print("  process the paper: " + self.name)


class Category:
    def __init__(self, folder_name):
        self.name = folder_name.split('.')[1].strip()
        print("process the category: " + self.name)
        path = os.path.join(folder_prefix, folder_name)
        files = os.listdir(path)
        self.papers = [Paper(file, path) for file in files if file != 'desktop.ini']
        print()


if __name__ == '__main__':
    t1 = time.time()
    icml_papers = os.listdir(folder_prefix)
    # all papers of ICML
    for sub_papers in icml_papers:
        key = sub_papers.split('.')[0].strip()
        categories[key] = Category(sub_papers)
    t2 = time.time()
    print("finished pre-processing %s papers,cost:%f s \n" % (total_papers_num, t2 - t1))
    # calculate idf
    for word in all_words:
        idf(word, categories)
    t3 = time.time()
    print("finished calculating IDF,cost:%f s \n" % (t3 - t2))
    # sort all words in the lexicographical order
    words_index = list(all_words)
    words_index.sort()
    # print vectors of target papers
    out_category(category_out)

    print("\noutput stopwords\n")
    out_stopwords()
    print("output words index")
    out_words_index()
