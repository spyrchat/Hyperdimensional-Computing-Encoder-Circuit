from HDC_library import *
import numpy as np
import matplotlib.pyplot as plt
import os

def test_F_of_x():

    #Parameters
    imgsize_vector = 30 #Each input vector has 30 features
    n_class = 2
    D_b = 4 #We target 4-bit HDC prototypes
    B_cnt = 8
    maxval = 256 #The input features will be mapped from 0 to 255 (8-bit)
    D_HDC = 100 #HDC hypervector dimension
    portion = 0.6 #We choose 60%-40% split between train and test sets
    Nbr_of_trials = 1 #Test accuracy averaged over Nbr_of_trials runs
    lambda_1 = 1
    lambda_2 = 0
        
    #Load dataset
    dataset_path = 'WISCONSIN/data.csv' 
    DATASET = np.loadtxt(dataset_path, dtype = object, delimiter = ',', skiprows = 1)
    X = DATASET[:,2:].astype(float)
    LABELS = DATASET[:,1]
    LABELS[LABELS == 'M'] = 1
    LABELS[LABELS == 'B'] = 2
    LABELS = LABELS.astype(float)
    X = X.T / np.max(X, axis = 1)
    X, LABELS = shuffle(X.T, LABELS)
    imgsize_vector = X.shape[1]
    N_train = int(X.shape[0]*portion)

    Simplex = np.load("Simplex2.npz", allow_pickle = True)['data']
    simp_arr = Simplex[0] #Take Simplex from list
    gamma = simp_arr[0] #Regularization hyperparameter
    alpha_sp = simp_arr[1] #Threshold of accumulators
    beta_ = simp_arr[2] #incrementation step of accumulators
    
    grayscale_table = lookup_generate(D_HDC, maxval, mode = 1) #Input encoding LUT
    position_table = lookup_generate(D_HDC, imgsize_vector, mode = 0) #weight for XOR-ing
    HDC_cont_all = np.zeros((X.shape[0], D_HDC)) #Will contain all "bundled" HDC vectors
    bias_ = np.random.randint(-2**(B_cnt-1), 2**(B_cnt-1)-1, (X.shape[0],D_HDC)) # -> INSERT YOUR CODE #generate the random biases once, [0,2*pi[ uniform

    for i in range(X.shape[0]): # for every patient
        if i%100 == 0:
            print(str(i) + "/" + str(X.shape[0]))
        HDC_cont_all[i,:] = encode_HDC_RFF(np.round((maxval - 1) * X[i,:]).astype(int), position_table, grayscale_table, D_HDC)
    
    # not quantized, test = train
    test_F_of_x_generic(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, False, True)

    # quantized, test = train
    test_F_of_x_generic(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, True, True)

    # quantized, test != train
    test_F_of_x_generic(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, True, False)

    # not quantized, test != train
    test_F_of_x_generic(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, False, False)

    return

def test_F_of_x_generic(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, quantization, train_is_test):

    local_avg, local_avgre, local_sparse = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, quantization, train_is_test)

    print("Results of {} test; {} train and test data: ".format("quantized" if quantization else "non-quantized", "same" if train_is_test else "separate"))
    print("local_avg={}".format(local_avg))
    print("local_avgre={}".format(local_avgre))
    print("local_sparse={}".format(local_sparse))

test_F_of_x()