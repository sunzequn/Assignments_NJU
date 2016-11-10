import numpy as np
from collections import Counter
import time

log_file = "log"
f = open(log_file, 'a')


def log(str1):
    f.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\ " + str1)
    f.write('\n')


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
        data_label.append(row_raw[len(row_raw) - 1])
    return np.matrix(data_raw, dtype=np.float).transpose(), data_label


def load_dis_mat(dis_file, r, l):
    t = time.time()
    mat = np.fromfile(dis_file, dtype=np.float)
    log("load dis file, cost : %f" % (time.time() - t))
    mat.shape = (r, l)
    return np.matrix(mat)


def l1_norm(mat, vec):
    vec_mat = np.tile(vec, mat.shape[1])
    l1 = np.linalg.norm(abs(mat - vec_mat), 1, axis=0)
    return l1


def l2_norm(mat, vec):
    vec_mat = np.tile(vec, mat.shape[1])
    l2 = np.linalg.norm(mat - vec_mat, 2, axis=0)
    return l2


def generate_dis_mat(data_mat, dis_func):
    t = time.time()
    mat = dis_func(data_mat, data_mat[:, 0])
    for i in range(data_mat.shape[1]):
        if i > 0:
            vec_dis = dis_func(data_mat, data_mat[:, i])
            mat = np.row_stack((mat, vec_dis))
    log("generate distance matrix, cost %f s." % (time.time() - t))
    return np.matrix(mat)


def remove_all(l1, l2):
    return list(set(l1) - set(l2))


def get_labels(labels):
    values = list(set(labels))
    values.sort()
    log("numbers of true labels are : {}".format(Counter(labels)))
    return len(labels), values


# associate each data point to the closest medoid
def cluster(medoids_index, dis_mat):
    mat = dis_mat[medoids_index, :]
    min_clusters = np.argsort(mat, axis=0)
    clusters = min_clusters[0, :]
    return clusters.tolist()[0]


def evaluate(clusters, k, labels):
    labels_num, labels_values = get_labels(labels)
    confusion_mat = np.matrix(np.zeros((len(labels_values), k)))
    for i in range(labels_num):
        confusion_mat[labels_values.index(labels[i]), clusters[i]] += 1
    purity = cal_purity(confusion_mat)
    gini = cal_gini(confusion_mat)
    return purity, gini


def cal_purity(confusion_mat):
    cluster_max = np.max(confusion_mat, axis=0)
    purity = np.sum(cluster_max) / np.sum(confusion_mat)
    return purity


def cal_gini(confusion_mat):
    m = np.sum(confusion_mat, axis=0)
    m = m.getA()
    g = 1 - np.sum(np.square(confusion_mat / m), axis=0)
    g = g.getA()
    gini = np.sum(g * m) / np.sum(m)
    return gini
