from utils import *
import time
import random
import copy
import sys

german_file = "german.txt"
mnist_file = "mnist.txt"
german_l1_dis_file = "german_l1_dis.bin"
mnist_l1_dis_file = "mnist_l1_dis.bin"


# calculate total cost of all clusters
def cal_total_cost(dis_mat, medoids):
    mat = dis_mat[medoids, :]
    min_mat = np.min(mat, axis=0)
    return np.sum(min_mat)


# Use a greedy search to find k medoids
def cal_k_medoids(dis_mat, k):
    n = dis_mat.shape[0]
    all_index = [i for i in range(n)]
    # randomly select k of the data points as the medoids
    medoids_index = random.sample(all_index, k)
    log("random medoids are : {}".format(medoids_index))
    # calculate total cost of these clusters
    total_cost = cal_total_cost(dis_mat, medoids_index)
    initial_medoids_index = copy.deepcopy(medoids_index)
    # For each medoid m, for each non-medoid data point o, swap them if total cost decreases.
    for medoid in initial_medoids_index:
        all_nonmedoids_index = remove_all(all_index, medoids_index)
        swap_index = -1
        for nonmedoids in all_nonmedoids_index:
            new_medoids_index = copy.deepcopy(medoids_index)
            new_medoids_index.remove(medoid)
            new_medoids_index.append(nonmedoids)
            temp_total_cost = cal_total_cost(dis_mat, new_medoids_index)
            if temp_total_cost < total_cost:
                total_cost = temp_total_cost
                swap_index = nonmedoids
        if swap_index > -1:
            medoids_index.remove(medoid)
            medoids_index.append(swap_index)
    medoids_index.sort()
    return medoids_index, total_cost


def k_medoids_main(data_mat, labels, k, sim_file, is_load_file=True, is_save_file=False):
    t = time.time()
    log("k_medoids begins")
    print("k_medoids begins:")
    if is_load_file:
        dis_mat = load_dis_mat(sim_file, data_mat.shape[1], data_mat.shape[1])
    else:
        dis_mat = generate_dis_mat(data_mat, l1_norm)
        if is_save_file:
            dis_mat.tofile(sim_file)
    for i in range(10):
        log("try %d begins:" % i)
        medoids_index, total_cost = cal_k_medoids(dis_mat, k)
        log("k medoids are : {} and total cost is {}".format(medoids_index, total_cost))
        clusters = cluster(medoids_index, dis_mat)
        log("numbers of cluster labels are : {}".format(Counter(clusters)))
        purity, gini = evaluate(clusters, k, labels)
        print("try %d: purity is %f, gini is %f." % (i, purity, gini))
        log("purity is %f, gini is %f.\n" % (purity, gini))
    print("k_medoids, total time : %f" % (time.time() - t))
    log("k_medoids, total time : %f\n" % (time.time() - t))


if __name__ == '__main__':
    is_load = False
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        is_load = True

    print("german:")
    log("german:")
    german_data_mat, german_labels = read_data(german_file)
    # for dataset german.txt, run the program from the beginning.
    k_medoids_main(german_data_mat, german_labels, 2, german_l1_dis_file, is_load_file=False, is_save_file=False)

    print("\nmnist:")
    log("mnist:")
    mnist_data_mat, mnist_labels = read_data(mnist_file)
    k_medoids_main(mnist_data_mat, mnist_labels, 10, mnist_l1_dis_file, is_load_file=is_load, is_save_file=not is_load)

    close()
