# -*- coding: utf-8 -*-

from utils04 import *
import time

dataset1_training = 'dataset1-a9a-training.txt'
dataset1_testing = 'dataset1-a9a-testing.txt'
dataset2_training = 'covtype-training.txt'
dataset2_testing = 'covtype-testing.txt'


def sgd(title, gradient, loss, training_mat, label, test_mat, test_label, iterations, rate, lam):
    log(title)
    m = 20
    print(title)
    data_mat = add_one(training_mat)
    f = data_mat.shape[0]  # f = the number of features + 1
    n = data_mat.shape[1]  # the number of examples
    beta = np.matrix(np.random.random((f, 1)))  # randomly generate parameters
    t = time.time()
    losses = []
    training_errors = []
    testing_errors = []
    iterationses = [i+1 for i in range(m * iterations)]
    for i in range(iterations):
        index = index_shuffle(n)
        num = 0
        it = int(n / m)
        for j in index:
            num += 1
            xj = data_mat[:, j]
            new_beta = beta - rate * gradient(beta, xj, label[j], lam)
            beta = new_beta
            if num % it == 0:
                lo = loss(data_mat, label, beta, lam)
                losses.append(lo)
                training_error = test(training_mat, label, beta)
                training_errors.append(training_error)
                testing_error = test(test_mat, test_label, beta)
                testing_errors.append(testing_error)
                log(str(lo) + '\t' + str(training_error) + '\t' + str(testing_error))
    plot(title, losses, training_errors, testing_errors, iterationses)
    print("cost : %f s\n" % (time.time() - t))


if __name__ == '__main__':
    iterations = 4
    lam = 0.0001
    dataset1_training_mat, dataset1_training_labels = read_data(dataset1_training)
    dataset1_testing_mat, dataset1_testing_labels = read_data(dataset1_testing)
    dataset2_training_mat, dataset2_training_labels = read_data(dataset2_training)
    dataset2_testing_mat, dataset2_testing_labels = read_data(dataset2_testing)

    sgd("logistic regression on dataset1", gradient_logistic, loss_logistic, dataset1_training_mat,
        dataset1_training_labels, dataset1_testing_mat, dataset1_testing_labels, iterations, 0.0001, lam)
    sgd("logistic regression on dataset2", gradient_logistic, loss_logistic, dataset2_training_mat,
        dataset2_training_labels, dataset2_testing_mat, dataset2_testing_labels, iterations, 0.00006, lam)

    sgd("ridge regression on dataset1", gradient_ridge, loss_ridge, dataset1_training_mat,
        dataset1_training_labels, dataset1_testing_mat, dataset1_testing_labels, iterations, 0.0001, lam)
    sgd("ridge regression on dataset2", gradient_ridge, loss_ridge, dataset2_training_mat,
        dataset2_training_labels, dataset2_testing_mat, dataset2_testing_labels, iterations, 0.00006, lam)

    close()
