# -*- coding: utf-8 -*-

from utils05 import *
from collections import Counter


def cal_prob_c(data_labels):
    total = len(data_labels)
    count = Counter(data_labels)
    for c in count.keys():
        count[c] = (count[c] + 1) / (total + 2)
    return count


def generate_k(x_index, c_value, total_mat, total_labels):
    if c_value is not None:
        c_index = np.where(total_labels == c_value)[0]
        total_mat_c_x = total_mat[c_index, x_index]
    else:
        total_mat_c_x = total_mat[:, x_index]
    values = set()
    for i in range(total_mat_c_x.shape[0]):
        values.add(total_mat_c_x[i, 0])
    k = len(values)
    return k


def cal_cond_prob(data_mat, data_flag, data_labels, x_index, x_value, c_value, total_mat, total_labels, cond_prop_map):
    key = str(x_index) + '_' + str(x_value) + '_' + str(c_value)
    if key in cond_prop_map.keys():
        return cond_prop_map[key]
    c_index = np.where(data_labels == c_value)[0]
    total_c = len(c_index)
    if len(c_index) == 0:
        return None
    data_mat_c_x = data_mat[c_index, x_index]
    # discrete feature
    if data_flag[x_index] == 1:
        total_x = len(np.where(data_mat_c_x == x_value)[0]) + 1
        k = generate_k(x_index, c_value, total_mat, total_labels)
        total_c += k
        cond_prop_map[key] = round(total_x / total_c, 8)
        return round(total_x / total_c, 8)
    # numerical feature
    else:
        mu = np.mean(data_mat_c_x)
        sigma = np.std(data_mat_c_x)
        cond_prop_map[key] = normal_distribution(x_value, mu, sigma)
        return normal_distribution(x_value, mu, sigma)


def cal_prob(data_mat, data_flag, x_index, x_value, total_mat, total_labels, prop_map):
    key = str(x_index) + '_' + str(x_value)
    if key in prop_map.keys():
        return prop_map[key]
    total = data_mat.shape[0]
    data_mat_x = data_mat[:, x_index]
    # discrete feature
    if data_flag[x_index] == 1:
        total_x = len(np.where(data_mat_x == x_value)[0]) + 1
        k = generate_k(x_index, None, total_mat, total_labels)
        total += k
        prop_map[key] = round(total_x / total, 8)
        return round(total_x / total, 8)
    # numerical feature
    else:
        mu = np.mean(data_mat_x)
        sigma = np.std(data_mat_x)
        prop_map[key] = normal_distribution(x_value, mu, sigma)
        return normal_distribution(x_value, mu, sigma)


def cal_prob_prediction_c(data_mat, data_labels, data_flag, vec, c, total_mat, total_labels, cond_prop_map, prop_map):
    prob_c = cal_prob_c(data_labels).get(c)
    prob_xs = 1
    prob_conds = 1
    for i in range(vec.shape[1]):
        prob_conds *= cal_cond_prob(data_mat, data_flag, data_labels, i, vec[0, i], c, total_mat, total_labels, cond_prop_map)
        prob_xs *= cal_prob(data_mat, data_flag, i, vec[0, i], total_mat, total_labels, prop_map)
    return prob_conds * prob_c / prob_xs


def cal_prob_prediction(data_mat, data_labels, data_flag, vec, cs, total_mat, total_labels, cond_prop_map, prop_map):
    res = 0
    prob = -1

    for c in cs:
        prob_c = cal_prob_prediction_c(data_mat, data_labels, data_flag, vec, c, total_mat, total_labels, cond_prop_map, prop_map)
        if prob_c > prob:
            res = c
            prob = prob_c
    return res


def pred(train_mat, train_labels, test_mat, data_flag, total_mat, total_labels):
    cond_prop_map = dict()
    prop_map = dict()
    labels = set(train_labels)
    pred_lables = []
    for i in range(test_mat.shape[0]):
        p = cal_prob_prediction(train_mat, train_labels, data_flag, test_mat[i, :], labels, total_mat, total_labels, cond_prop_map, prop_map)
        pred_lables.append(p)
    return np.array(pred_lables)


if __name__ == '__main__':
    cross_n = 10
    # data_mat, data_labels, data_flag = read_data("german-assignment5.txt")
    data_mat, data_labels, data_flag = read_data("breast-cancer-assignment5.txt")
    train_n = int(data_mat.shape[0] / cross_n) * (cross_n - 1)
    train_indexes, test_indexes = cross_validation_data(data_mat.shape[0], train_n)
    train_mat, train_labels = data_mat[train_indexes, :], data_labels[train_indexes]
    test_mat, test_labels = data_mat[test_indexes, :], data_labels[test_indexes]
    pred_lables = pred(train_mat, train_labels, test_mat, data_flag, train_mat, train_labels)
    print(cal_acc(test_labels, pred_lables))
