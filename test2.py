from HDC_library import *
import numpy as np


def test_train_HDC_RFF():
    n_class = 2
    N_train = 360
    gamma = 0.2* 1e-5 # Choose arbitrary value
    D_b = 4
    # make some test data with one half are 1s and the other half are -1s
    Y_train_init = np.concatenate((np.ones(N_train),np.ones(N_train)*(-1)))
    HDC_cont_train = np.concatenate((np.ones((N_train,100)),np.ones((N_train,100))*(-1)))
    centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, 2*N_train, Y_train_init, HDC_cont_train, gamma, D_b)
    Acc = compute_accuracy(HDC_cont_train, Y_train_init, centroids_q, biases_q)
    assert Acc == 1.0
    print("Test training -all ones- OK")


def test_train_HDC_RFF_with_random_data():
    n_class = 2
    N_train = 360
    N_test = 360
    gamma = 0.2   # Arbitrary small value
    D_b = 4

    # Generate random training and testing data
    Y_train = np.random.choice([-1, 1], N_train)
    Y_test = np.random.choice([-1, 1], N_test)
    HDC_cont_train = np.random.rand(N_train, 100)
    HDC_cont_test = np.random.rand(N_test, 100)

    # Train the HDC model
    centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train, gamma, D_b)

    # Evaluate the model
    Acc_train = compute_accuracy(HDC_cont_train, Y_train, centroids_q, biases_q)
    Acc_test = compute_accuracy(HDC_cont_test, Y_test, centroids_q, biases_q)

    print(f"Training Accuracy on Random Data: {Acc_train}")
    print(f"Testing Accuracy on Random Data: {Acc_test}")

# test_train_HDC_RFF_with_random_data()


def test_train_HDC_RFF_on_wisconsin():
    # Load dataset
    dataset_path = 'WISCONSIN/data.csv'
    DATASET = np.loadtxt(dataset_path, dtype=object, delimiter=',', skiprows=1)
    X = DATASET[:, 2:].astype(float)
    LABELS = DATASET[:, 1]
    LABELS[LABELS == 'M'] = 1
    LABELS[LABELS == 'B'] = -1
    LABELS = LABELS.astype(float)
    X = X.T / np.max(X, axis=1)
    X, LABELS = shuffle(X.T, LABELS)

    # Parameters
    n_class = 2
    portion = 0.6  # 60% of data for training
    N_train = int(X.shape[0] * portion)  # Number of training samples
    gamma = 0.2   # Example value
    D_b = 4

    # Split data into training and testing sets
    HDC_cont_train = X[:N_train]
    LABELS_train = LABELS[:N_train]
    HDC_cont_test = X[N_train:]
    LABELS_test = LABELS[N_train:]

    # Train the HDC model
    centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, LABELS_train, HDC_cont_train, gamma, D_b)

    # Evaluate the model on the training set
    Acc_train = compute_accuracy(HDC_cont_train, LABELS_train, centroids_q, biases_q)
    print(f"Training Accuracy on Wisconsin Dataset: {Acc_train}")

    # Evaluate the model on the testing set
    Acc_test = compute_accuracy(HDC_cont_test, LABELS_test, centroids_q, biases_q)
    print(f"Testing Accuracy on Wisconsin Dataset: {Acc_test}")

test_train_HDC_RFF_on_wisconsin()




