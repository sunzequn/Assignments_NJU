# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import matplotlib.pyplot as plt

log_file = "results.txt"
f = open(log_file, 'w')


def log(str1):
    f.write(str1 + '\n')


def close():
    f.close()


# Read training and testing data to matrices.
# Each column is a example.
def read_data(data_file):
    data_raw = []
    data_label = []
    file = open(data_file, 'r')
    for line in file:
        row_raw = line.strip('\n').split(',')
        data_raw.append(row_raw[0:len(row_raw) - 1])
        data_label.append(float(row_raw[len(row_raw) - 1]))
    return np.matrix(data_raw, dtype=np.float).transpose(), data_label


def index_shuffle(n, m=0):
    index = [i for i in range(n)]
    random.shuffle(index)
    if m > 0:
        return random.sample(index, m)
    return index


def test(test_mat, labels, beta):
    test_mat = add_one(test_mat)
    total = len(labels)
    right = 0
    for i in range(len(labels)):
        xi = test_mat[:, i]
        yi = (beta.transpose() * xi)[0, 0]
        if yi * float(labels[i]) > 0:
            right += 1
    error_rate = 1 - right / total
    return error_rate


def add_one(data_mat):
    one_mat = np.matrix(np.ones((1, data_mat.shape[1])))
    return np.row_stack((data_mat, one_mat))


def gradient_logistic(beta, xi, yi, lam):
    e = beta.transpose().dot(xi)[0, 0] * yi
    e = math.exp(e) + 1
    return (-yi / e) * xi + lam * np.sum(np.sign(beta))


def loss_logistic(data_mat, labels, beta, lam):
    n = data_mat.shape[1]
    labels = np.matrix(labels)
    e = np.array(-labels) * np.array(beta.transpose().dot(data_mat))
    e = np.matrix(e)
    total = np.sum(np.log(1 + np.power(math.e, e)))
    return total / n + lam * np.sum(abs(beta))


def gradient_ridge(beta, xi, yi, lam):
    return 2 * (beta.transpose().dot(xi)[0, 0] - yi) * xi + 2 * lam * beta


def loss_ridge(data_mat, labels, beta, lam):
    n = data_mat.shape[1]
    labels = np.matrix(labels)
    total = np.sum(np.square(labels - beta.transpose().dot(data_mat)))
    return total / n + np.linalg.norm(beta, 2) * lam


def plot(title, losses, training_errors, test_errors, iterations):
    plt.plot(iterations, losses, 'b-', label='loss')
    plt.plot(iterations, training_errors, 'm:', label='training errors')
    plt.plot(iterations, test_errors, 'k-.', label='testing errors')
    plt.title(title)
    plt.xlabel('iterations')
    plt.xlim(1, len(iterations))
    plt.ylim(0, 1.0)
    plt.ylabel('error rate')
    plt.legend()
    plt.savefig(title)
    plt.close()

