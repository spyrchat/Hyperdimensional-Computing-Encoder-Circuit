from HDC_library import *
import numpy as np

def test_train_HDC_RFF():
    n_class = 2
    N_train = 360
    gamma = 0.0002 # Choose arbitrary value
    D_b = 4
    # make some test data with one half are 1s and the other half are -1s
    Y_train_init = np.concatenate((np.ones(N_train),np.ones(N_train)*(-1)))
    HDC_cont_train = np.concatenate((np.ones((N_train,100)),np.ones((N_train,100))*(-1)))
    centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, 2*N_train, Y_train_init, HDC_cont_train, gamma, D_b)
    Acc = compute_accuracy(HDC_cont_train, Y_train_init, centroids_q, biases_q)
    assert Acc == 1.0
    print("Test training OK")

def test_evaluate_F_of_x():
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
    bias_ = np.random.randint(0, 2**(B_cnt)-1, (X.shape[0],D_HDC)) #generate the random biases once

    for i in range(X.shape[0]): # for every patient
        if i%100 == 0:
            print(str(i) + "/" + str(X.shape[0]))
        HDC_cont_all[i,:] = encode_HDC_RFF(np.round((maxval - 1) * X[i,:]).astype(int), position_table, grayscale_table, D_HDC)
    
    # not quantized, test = train
    helper_evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, False, True)

    # quantized, test = train
    helper_evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, True, True)

    # not quantized, test != train
    helper_evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, False, False)

    # quantized, test != train
    helper_evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, True, False)

    return

def helper_evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, quantization, train_is_test):

    local_avg, local_avgre, local_sparse = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt, quantization, train_is_test)

    print("Results of {} test; {} train and test data: ".format("quantized" if quantization else "non-quantized", "equal" if train_is_test else "different"))
    print("local_avg={}".format(local_avg))
    print("local_avgre={}".format(local_avgre))
    print("local_sparse={}".format(local_sparse))

test_train_HDC_RFF()
test_evaluate_F_of_x()
