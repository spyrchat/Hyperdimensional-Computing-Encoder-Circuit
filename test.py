from HDC_library import train_HDC_RFF
from HDC_library import compute_accuracy
import numpy as np

def test_train():
    n_class = 2
    N_train = 360
    gamma = 0.0002
    D_b = 4
    Y_train_init = np.concatenate((np.ones(N_train),np.ones(N_train)*(-1)))
    HDC_cont_train = np.concatenate((np.ones((N_train,100)),np.ones((N_train,100))*(-1)))
    centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, 2*N_train, Y_train_init, HDC_cont_train, gamma, D_b)
    # print("centroids =",centroids)
    # print("biases =",biases)
    Acc = compute_accuracy(HDC_cont_train, Y_train_init, centroids_q, biases_q)
    return Acc

Acc = test_train()
print("Acc =",Acc)