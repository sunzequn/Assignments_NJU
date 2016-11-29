# -*- coding: utf-8 -*-

from naive_bayes_classifier import *
import copy


def init_w(n):
    data_w = []
    for i in range(n):
        data_w.append(round(1 / n, 8))
    return np.array(data_w)


# def update_w(w, w_indexes, labels, pred_labels):
#     e = 0
#     num = 0
#     for i in range(len(labels)):
#         if pred_labels[i] != labels[i]:
#             e += w[w_indexes[i]]
#             num += 1
#     if e == 0 or e >= 0.5:
#         return w, 0, 0
#     alpha = 0.5 * math.log((1 - e) / e)
#     z = 0
#     for i in range(len(labels)):
#         z += (w[w_indexes[i]] * math.exp(-alpha * if_equal(pred_labels[i], labels[i])))
#     for i in range(len(labels)):
#         w[w_indexes[i]] = (w[w_indexes[i]] / z) * math.exp(-alpha * if_equal(pred_labels[i], labels[i]))
#     return w, e, alpha


def update_w(w, labels, pred_labels):
    e = 0
    num = 0
    for i in range(len(labels)):
        if pred_labels[i] != labels[i]:
            e += w[i]
            num += 1
    if e == 0.0 or e >= 0.5:
        return w, e, 0
    alpha = 0.5 * math.log((1 - e) / e)
    z = 0
    for i in range(len(labels)):
        z += (w[i] * math.exp(-alpha * if_equal(pred_labels[i], labels[i])))
    for i in range(len(labels)):
        w[i] = (w[i] / z) * math.exp(-alpha * if_equal(pred_labels[i], labels[i]))
    return w, e, alpha


def weight_sample(data_w):
    total = np.sum(data_w)
    ordered_w_index = np.argsort(data_w)
    random_value = random.uniform(np.min(data_w), total)
    sum_total = 0
    for w_index in ordered_w_index:
        sum_total += data_w[w_index]
        if random_value < sum_total:
            return w_index
    return None


def adaboost(data_mat, data_labels, data_flag, iteration_n, cross_n):
    train_n = int(data_mat.shape[0] / cross_n) * (cross_n - 1)
    train_indexes, test_indexes = cross_validation_data(data_mat.shape[0], train_n)
    train_mat, train_labels = data_mat[train_indexes, :], data_labels[train_indexes]
    train_w = init_w(train_n)
    test_mat, test_labels = data_mat[test_indexes, :], data_labels[test_indexes]
    train_mat_list = []
    train_labels_list = []
    alpha_list = []
    for i in range(iteration_n):
        if i > 0:
            inner_train_indexes = []
            for j in range(train_n):
                inner_train_indexes.append(weight_sample(train_w))
            inner_train_indexes = np.array(inner_train_indexes)
            inner_train_mat = train_mat[inner_train_indexes, :]
            inner_train_labels = train_labels[inner_train_indexes]
        else:
            inner_train_mat = train_mat
            inner_train_labels = train_labels

        train_mat_list.append(copy.deepcopy(inner_train_mat))
        train_labels_list.append(copy.deepcopy(inner_train_labels))

        pred_labels = pred(inner_train_mat, inner_train_labels, train_mat, data_flag, inner_train_mat, inner_train_labels)
        train_w, e, alpha = update_w(train_w, train_labels, pred_labels)
        print("iteration %d : e = %f, alpha = %f" % (i, e, alpha))
        alpha_list.append(alpha)
        if abs(alpha - 0) < 0.0000001:
            break
    return test(train_mat_list, train_labels_list, alpha_list, test_mat, test_labels, data_flag, train_mat, train_labels)


def test(train_mat_list, train_labels_list, alpha_list, test_mat, test_labels, data_flag, total_mat, total_labels):
    labels = pred(train_mat_list[0], train_labels_list[0], test_mat, data_flag, total_mat, total_labels)
    acc = cal_acc(test_labels, labels)
    print("naive acc:", acc)
    labels = labels * alpha_list[0]
    for i in range(1, len(train_mat_list)):
        pred_labels = pred(train_mat_list[i], train_labels_list[i], test_mat, data_flag, total_mat, total_labels)
        pred_labels = pred_labels * alpha_list[i]
        labels += pred_labels
    for i in range(len(labels)):
        labels[i] = np.sign(labels[i])
    acc = cal_acc(test_labels, labels)
    print("adaboost acc = %f" % acc)
    return acc


def adaboost_main(data_mat, data_labels, data_flag, inner_loop, outer_loop):
    acc = []
    for i in range(10):
        print("times %d: " % i)
        acc.append(adaboost(data_mat, data_labels, data_flag, inner_loop, outer_loop))
        print()
    acc = np.array(acc)
    print("average acc: %f, sdt: %f \n" % (np.mean(acc), np.std(acc)))


if __name__ == '__main__':
    inner_loop = 10
    outer_loop = 10

    data_mat1, data_labels1, data_flag1 = read_data("breast-cancer-assignment5.txt")
    print("breast-cancer-assignment5.txt:")
    adaboost_main(data_mat1, data_labels1, data_flag1, inner_loop, outer_loop)

    data_mat2, data_labels2, data_flag2 = read_data("german-assignment5.txt")
    print("german-assignment5.txt")
    adaboost_main(data_mat2, data_labels2, data_flag2, inner_loop, outer_loop)
