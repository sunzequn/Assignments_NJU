# -*- coding: utf-8 -*-

import numpy as np
import math
import random


def read_data(data_file):
    '''
    Read data to matrices. Each row is a example.
    :param data_file:
    :return:
    '''
    data_raw = []
    data_label = []
    data_flag = []
    file = open(data_file, 'r')
    lines = file.readlines()
    for i in range(len(lines)):
        line = lines[i]
        row_raw = line.strip('\n').split(',')
        if i == 0:
            for f in row_raw:
                data_flag.append(int(f))
        else:
            data_raw.append(row_raw[0:len(row_raw) - 1])
            data_label.append(pre(float(row_raw[len(row_raw) - 1])))
    return np.matrix(data_raw, dtype=np.float), np.array(data_label), np.array(data_flag)


def pre(n):
    if n < 1:
        return -1
    return n


def cross_validation_data(data_num, n):
    indexes = [i for i in range(data_num)]
    train_indexes = random.sample(indexes, n)
    test_indexes = list(set(indexes).difference(set(train_indexes)))
    return train_indexes, test_indexes


def normal_distribution(x, mu, sigma):
    return round((1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-math.pow(x - mu, 2) / (2 * math.pow(sigma, 2))), 8)


def cal_acc(labels, pred_labels):
    acc_num = 0
    for i in range(len(labels)):
        if labels[i] == pred_labels[i]:
            acc_num += 1
    acc = acc_num / len(labels)
    return acc
