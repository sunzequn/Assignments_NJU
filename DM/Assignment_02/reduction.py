# encoding=utf-8
import numpy as np
from collections import Counter
from itertools import product
from dijkstar import Graph
from dijkstar.algorithm import single_source_shortest_paths
import time

sonar_test = './sonar-test.txt'
sonar_train = './sonar-train.txt'
splice_test = './splice-test.txt'
splice_train = './splice-train.txt'
max_dis = float("inf")
isomap_k = 12

"""
Use a property of Laplacian matrix to judge weather the graph is connected.
The property is that The number of times 0 appears as an eigenvalue in the Laplacian
is the number of connected components in the graph.
"""
def is_connected(graph_mat, max_dis):
    n = graph_mat.shape[0]
    adjacency_mat = np.tile(np.matrix(np.tile(0, n)).transpose(), n)

    def f(x, y):
        adjacency_mat[x, y] = 1

    list((f(i, j) for i, j in product(range(n), range(n)) if graph_mat[i, j] != max_dis and i != j))
    degree = np.sum(adjacency_mat, axis=1).transpose().tolist()[0]
    degree_mat = np.diag(degree)
    laplacian_matrix = degree_mat - adjacency_mat
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    zeros = [e for e in eigenvalues if abs(e) < 1e-10]
    return True if len(zeros) == 1 else False


# Get the k nearest points
def knn_point(train_mat, test_vec, k):
    test_mat = np.tile(test_vec, train_mat.shape[1])
    dis_mat = np.sqrt(np.square(train_mat - test_mat).sum(axis=0))
    dis_min_k = np.argsort(dis_mat).tolist()[0][0:k]
    return dis_min_k, dis_mat


# Implementation of K-NN
def knn_label(train_mat, train_labels, test_vec, k):
    dis_min_top_k = knn_point(train_mat, test_vec, k)[0]
    labels = [train_labels[i] for i in dis_min_top_k]
    if k == 1:
        return labels[0]
    count_labels = Counter(labels)
    order_labels = sorted(count_labels.items(), key=lambda d: d[1])
    return order_labels[len(order_labels) - 1][0]


# Calculate the accuracy of predicting the label of testing examples
def train_test(train_mat_low, train_label, test_mat_low, test_label, knn_k, top_k):
    total_test = test_mat_low.shape[1]
    right = 0
    for i in range(total_test):
        label_pre = knn_label(train_mat_low, train_label, test_mat_low[:, i], knn_k)
        if label_pre == test_label[i]:
            right += 1
    if top_k > 0:
        print("k = %d and the accuracy is %d / %d = %f" % (top_k, right, total_test, right / total_test))
    else:
        print("the accuracy is %d / %d = %f" % (right, total_test, right / total_test))


# Read training and testing data to matrices
def read_data(data_file):
    data_raw = []
    data_label = []
    file = open(data_file, 'r')
    for line in file:
        row_raw = line.strip('\n').split(',')
        data_raw.append(row_raw[0:len(row_raw) - 1])
        data_label.append(row_raw[len(row_raw) - 1])
    return np.matrix(data_raw, dtype=np.float).transpose(), data_label


# Use the data to learn the projection matrix.
def pca_pro_mat(data_mat, k):
    # remove means
    mean_vec = np.mean(data_mat, axis=1)
    # print(mean_vec.shape)
    data_mat = data_mat - mean_vec
    # calculate the covariance matrix
    covariance_mat = np.cov(data_mat)
    # calculate the eigenvalues and the eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
    # calculate the k largest eigenvectors
    eigenvalues_top_k = np.argsort(-eigenvalues)[0:k]
    eigenvectors_top_k = eigenvectors[:, eigenvalues_top_k]
    return eigenvectors_top_k


# Project the data and to the lower-dimensional space via the projection matrix
def pca(data_mat, pro_mat):
    # data_mat_low = (data_mat.transpose() * pro_mat).transpose()
    # print(pro_mat.shape, data_mat.shape)
    data_mat_low = pro_mat.transpose() * data_mat
    # print(data_mat_low.shape)
    return data_mat_low


# Main method for testing PCA
def pca_main(train_mat, train_label, test_mat, test_label, ks):
    t1 = time.time()
    for k in ks:
        pro_mat = pca_pro_mat(train_mat, k)
        train_mat_low = pca(train_mat, pro_mat)
        test_mat_low = pca(test_mat, pro_mat)
        train_test(train_mat_low, train_label, test_mat_low, test_label, 1, k)
    print('cost: %f s\n' % (time.time() - t1))


# Use the data to learn the projection matrix.
def svd_w(data_mat, k):
    # use the existing tools directly to conduct SVD of a matrix
    u, s, v = np.linalg.svd(data_mat)
    return u[:, 0:k]


# Project the data and to the lower-dimensional space via the projection matrix
def svd(data_mat, w):
    return w.transpose() * data_mat


# Main method for testing SVD
def svd_main(train_mat, train_label, test_mat, test_label, ks):
    t1 = time.time()
    for k in ks:
        w = svd_w(train_mat, k)
        train_mat_low = svd(train_mat, w)
        test_mat_low = svd(test_mat, w)
        train_test(train_mat_low, train_label, test_mat_low, test_label, 1, k)
    print('cost: %f s\n' % (time.time() - t1))


# Calculate the k that makes the graph connected.
def min_k_connected(train_mat, test_mat):
    k = 4
    while True:
        mat = graph_knn(train_mat, test_mat, k)
        if is_connected(mat, max_dis):
            print('k = %d, the graph is connected.' % k)
            return k
        k += 1


# Use K-NN to construct a weighted graph
def graph_knn(train_mat, test_mat, k):
    data_mat = np.column_stack((train_mat, test_mat))
    n = data_mat.shape[1]
    mat = np.matrix(np.tile(max_dis, n)).transpose()
    mat = np.tile(mat, n)
    for i in range(data_mat.shape[1]):
        top_k_points, dis_mat = knn_point(data_mat, data_mat[:, i], k + 1)
        for j in top_k_points:
            mat[i, j] = dis_mat[0, j]
            mat[j, i] = dis_mat[0, j]
    return mat


# Generate the distance matrix
def dist_mat(train_mat, test_mat, k):
    # construct a weighted graph
    mat = graph_knn(train_mat, test_mat, k)
    n = mat.shape[0]
    graph = Graph()
    list((graph.add_edge(i, j, {'cost': mat[i, j]}) for i, j in product(range(n), range(n)) if
          i != j and mat[i, j] != max_dis))
    if graph is None:
        return
    cost_func = lambda u, v, e, prev_e: e['cost']
    mat = np.zeros((n, n))
    # the shortest path from node i to node j is the distance between i and j
    def dis(i):
        single_short_path = single_source_shortest_paths(graph, i, cost_func=cost_func)
        for j in range(n):
            if j != i:
                mat[i, j] = extract_shortest_path(single_short_path, j)
            else:
                mat[i, j] = 0
    list((dis(i) for i in range(n)))
    return mat


# extract shortest path from the results calculated by 'dijkstar'
def extract_shortest_path(predecessors, d):
    # costs of the edges on the shortest path from s to d
    costs = []
    try:
        u, e, cost = predecessors[d]
        while u is not None:
            costs.append(cost)
            u, e, cost = predecessors[u]
        costs.reverse()
        return sum(costs)
    except Exception:
        print('The graph is not connected')
        exit()


# Eigen decompose
def isomap_eig(mat):
    n = mat.shape[0]
    # the squared proximity matrix
    mat2 = np.square(mat)
    # double centering
    c = np.eye(n) - 1 / n
    # calculate the dot-product matrix
    b = -0.5 * c.dot(mat2).dot(c)
    eigenvalues, eigenvectors = np.linalg.eig(b)
    return eigenvalues, eigenvectors


# Calculate the lower-dimensional matrix
def isomap_w(eigenvalues, eigenvectors, k):
    eigenvalues_top_k_index = np.argsort(-eigenvalues)[0:k]
    # the k largest eigenvalues and eigenvectors
    eigenvalues_top_k = eigenvalues[eigenvalues_top_k_index]
    eigenvectors_top_k = eigenvectors[:, eigenvalues_top_k_index]
    w = eigenvectors_top_k.dot(np.diag(np.sqrt(eigenvalues_top_k)))
    return w.transpose()


# Main method for testing ISOMAP
def isomap_main(train_mat, train_label, test_mat, test_label, ks):
    t1 = time.time()
    mat = dist_mat(train_mat, test_mat, isomap_k)
    eigenvalues, eigenvectors = isomap_eig(mat)
    for k in ks:
        w = isomap_w(eigenvalues, eigenvectors, k)
        train_n = train_mat.shape[1]
        test_n = test_mat.shape[1]
        train_mat_low = np.matrix(w[:, 0: train_n])
        test_mat_low = np.matrix(w[:, train_n: train_n + test_n])
        train_test(train_mat_low, train_label, test_mat_low, test_label, 1, k)
    print('cost: %f s\n' % (time.time() - t1))


if __name__ == '__main__':
    sonar_train_mat, sonar_train_label = read_data(sonar_train)
    sonar_test_mat, sonar_test_label = read_data(sonar_test)
    splice_train_mat, splice_train_label = read_data(splice_train)
    splice_test_mat, splice_test_label = read_data(splice_test)

    # min_k_connected(sonar_train_mat, sonar_test_mat)
    # min_k_connected(splice_train_mat, splice_test_mat)

    print('No reduction:')
    print('sonar:')
    train_test(sonar_train_mat, sonar_train_label, sonar_test_mat, sonar_test_label, 1, 0)
    print('splice:')
    train_test(splice_train_mat, splice_train_label, splice_test_mat, splice_test_label, 1, 0)

    print('\nPCA:')
    print('sonar:')
    pca_main(sonar_train_mat, sonar_train_label, sonar_test_mat, sonar_test_label, (10, 20, 30))
    print('splice:')
    pca_main(splice_train_mat, splice_train_label, splice_test_mat, splice_test_label, (10, 20, 30))

    print('\nSVD:')
    print('sonar:')
    svd_main(sonar_train_mat, sonar_train_label, sonar_test_mat, sonar_test_label, (10, 20, 30))
    print('splice:')
    svd_main(splice_train_mat, splice_train_label, splice_test_mat, splice_test_label, (10, 20, 30))

    print('\nISOMAP:')
    print('sonar:')
    isomap_main(sonar_train_mat, sonar_train_label, sonar_test_mat, sonar_test_label, (10, 20, 30))
    print('splice:')
    isomap_main(splice_train_mat, splice_train_label, splice_test_mat, splice_test_label, (10, 20, 30))

