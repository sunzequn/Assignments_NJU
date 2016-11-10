from k_medoids import *
import time
import scipy.sparse.linalg as l


german_file = "german.txt"
mnist_file = "mnist.txt"
german_l2_dis_file = "german_l2_dis.bin"
mnist_l2_dis_file = "mnist_l2_dis.bin"


# Get the k nearest points of i
def knn_point(dis_mat, i, k):
    dis_mat_i = dis_mat[i, :]
    dis_min_k = np.argsort(dis_mat_i).tolist()[0][0:k]
    return dis_min_k


# Use K-NN to construct a graph, return the adjacency matrix and the degree matrix of the graph
def graph_knn(dis_mat, k):
    t = time.time()
    n = dis_mat.shape[1]
    adjacency_mat = np.zeros((n, n))
    for i in range(n):
        nearest_k_points = knn_point(dis_mat, i, k + 1)
        for j in nearest_k_points:
            if i != j:
                # the weight of an edge is 1
                adjacency_mat[i, j] = adjacency_mat[j, i] = 1
    degree = np.sum(adjacency_mat, axis=0)
    degree_mat = np.diag(degree)
    log("knn graph, cost : %f" % (time.time() - t))
    return adjacency_mat, degree_mat


# calcualte the laplacian matrix and the smallest n eigenvectors of it
def laplacian(adjacency_mat, degree_mat, n):
    laplacian_matrix = degree_mat - adjacency_mat
    t = time.time()
    eigenvalues, eigenvectors = l.eigsh(laplacian_matrix, k=(n+1), which='SA')
    log("generate eigs, cost : %f" % (time.time() - t))
    zeros = [e for e in eigenvalues if abs(e) < 1e-10]
    if len(zeros) == 1:
        eigenvectors = np.delete(eigenvectors, 0, 1)
        return np.matrix(eigenvectors).transpose()
    else:
        log("the graph is not connected")
        return None


def spectral_main(data_mat, labels, n, k, dis_file, is_load_file=True, is_save_file=False):
    print("spectral clustering begins:")
    log("spectral clustering begins:")
    t = time.time()
    print("k = %d " % k)
    if is_load_file:
        dis_mat = load_dis_mat(dis_file, data_mat.shape[1], data_mat.shape[1])
    else:
        dis_mat = generate_dis_mat(data_mat, l2_norm)
    if is_save_file:
         dis_mat.tofile(dis_file)
    adjacency_mat, degree_mat = graph_knn(dis_mat, k)
    mat = laplacian(adjacency_mat, degree_mat, n)
    k_medoids_main(mat, labels, n, None, is_load_file=False)
    print("spectral clustering, total time : %f\n" % (time.time() - t))


if __name__ == '__main__':

    is_load = False
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        is_load = True

    print("german:")
    log("german:")
    german_data_mat, german_labels = read_data(german_file)
    spectral_main(german_data_mat, german_labels, 2, 3, german_l2_dis_file, is_load_file=False, is_save_file=False)
    spectral_main(german_data_mat, german_labels, 2, 6, german_l2_dis_file, is_load_file=False, is_save_file=False)
    spectral_main(german_data_mat, german_labels, 2, 9, german_l2_dis_file, is_load_file=False, is_save_file=False)

    print("\nmnist:")
    log("mnist:")
    mnist_data_mat, mnist_labels = read_data(mnist_file)
    spectral_main(mnist_data_mat, mnist_labels, 10, 3, mnist_l2_dis_file, is_load_file=is_load, is_save_file=not is_load)
    spectral_main(mnist_data_mat, mnist_labels, 10, 6, mnist_l2_dis_file, is_load_file=True, is_save_file=False)
    spectral_main(mnist_data_mat, mnist_labels, 10, 9, mnist_l2_dis_file, is_load_file=True, is_save_file=False)

    close()
