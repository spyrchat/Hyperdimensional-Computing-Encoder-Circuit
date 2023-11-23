"""
Design of a Hyperdimensional Computing Circuit for Bio-signal Classification via Nelder-Mead optimization
and LS-SVM Training.

*HDC library*

Computer-Aided IC Design (B-KUL-H05D7A)

ir. Ali Safa, ir. Sergio Massaioli, Prof. Georges Gielen (MICAS-IMEC-KU Leuven)

(Author: A. Safa)
"""

import numpy as np
from sklearn.utils import shuffle

# Receives the HDC encoded test set "HDC_cont_test" and test labels "Y_test"
# Computes test accuracy w.r.t. the HDC prototypes (centroids) and the biases found at training time
def compute_accuracy(HDC_cont_test, Y_test, centroids, biases):
    Acc = 0
    n_class = np.max(Y_test) + 1
    for i in range(Y_test.shape[0]):
        received_HDC_vector = HDC_cont_test[i]
        all_resp = np.zeros(n_class)
        for cl in range(n_class):
            final_HDC_centroid = centroids[cl]
             #compute LS-SVM response
            response = # -> INSERT YOUR CODE 
            all_resp[cl] = response
        
        class_idx = np.argmax(all_resp)
        if class_idx == Y_test[i]:
            Acc += 1
            
    return Acc/Y_test.shape[0]


# Generates random binary matrices of -1 and +1
# when mode == 0, all elements are generated with the same statistics (probability of having +1 always = 0.5)
# when mode == 1, all the probability of having +1 scales with an input "key" i.e., when the inputs to the HDC encoded are coded
# on e.g., 8-bit, we have 256 possible keys
def lookup_generate(dim, n_keys, mode = 1):
    if mode == 0:
        # -> INSERT YOUR CODE
    else:
        # -> INSERT YOUR CODE
        
    return table.astype(np.int8)
    

# Performs "part" of the HDC encoding (only input encoding, position encoding and bundling), without the thresholding at the end.
# Returns H = bundle_along_features(P.L)
# img is the input feature vector to be encoded
# position_table is the random matrix of mode == 0
# grayscale_table is the input encoding LUT of mode == 1
# dim is the HDC dimensionality D
def encode_HDC_RFF(img, position_table, grayscale_table, dim):
    img_hv = np.zeros(dim, dtype=np.int16)
    container = np.zeros((len(position_table), dim))
    for pixel in range(len(position_table)):
        #Get the input-encoding and XOR-ing result:  
        hv = # -> INSERT YOUR CODE
        container[pixel, :] = hv*1
        
    img_hv = np.sum(container, axis = 0) #bundling without the cyclic step yet
    return img_hv


# Train the HDC circuit on the training set : (Y_train, HDC_cont_train)
# n_class: number of clases
# N_train: number of data points in training set
# gamma: LS-SVM regularization
# D_b: number of bit for HDC prototype quantization
def train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train, gamma, D_b):
    centroids = []
    centroids_q = []
    biases_q = []
    biases = []
    for cla in range(n_class):
        #The steps below implement the LS-SVM training, check out the course notes, we are just implementing that
        #Beta.alpha = L -> alpha (that we want) 
        Beta = np.zeros((N_train+1, N_train+1)) #LS-SVM regression matrix
        #Fill Beta:
        
        # -> INSERT YOUR CODE
        
        #Target vector L:
            
        # -> INSERT YOUR CODE
        
        #Solve the system of equations to get the vector alpha:
            
        alpha = # -> INSERT YOUR CODE
        
        # Get HDC prototype for class cla, still in floating point
        
        final_HDC_centroid = # -> INSERT YOUR CODE
        
        # Quantize HDC prototype to D_b-bit
        final_HDC_centroid_q = # -> INSERT YOUR CODE 
        #Amplification factor for the LS-SVM bias
        fact = # -> INSERT YOUR CODE  
        if np.max(np.abs(final_HDC_centroid)) == 0:
            print("Kernel matrix badly conditionned! Ignoring...")
            centroids_q.append(np.ones(final_HDC_centroid_q.shape)) #trying to manage badly conditioned matrices, do not touch
            biases_q.append(10000)
        else:
            centroids_q.append(final_HDC_centroid_q*1)
            biases_q.append(alpha[0]*fact)
            
        centroids.append(final_HDC_centroid*1)
        biases.append(alpha[0])
        
    return centroids, biases, centroids_q, biases_q


# Evaluate the Nelder-Mead cost F(x) over "Nbr_of_trials" trials
# (HDC_cont_all, LABELS) is the complete dataset with labels
# beta_ is the output accumulator increment of the HDC encoder
# bias_ are the random starting value of the output accumulators
# gamma is the LS-SVM regularization hyper-parameter
# alpha_sp is the encoding threshold
# n_class is the number of classes, N_train is the number training points, D_b the HDC prototype quantization bit width
# lambda_1, lambda_2 define the balance between Accuracy and Sparsity: it returns lambda_1*Acc + lambda_2*Sparsity
def evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt):
    local_avg = np.zeros(Nbr_of_trials)
    local_avgre = np.zeros(Nbr_of_trials)
    local_sparse = np.zeros(Nbr_of_trials)
    #Estimate F(x) over "Nbr_of_trials" trials
    for trial_ in range(Nbr_of_trials): 
        HDC_cont_all, LABELS = shuffle(HDC_cont_all, LABELS) # Shuffle dataset for random train-test split
            
        HDC_cont_train_ = HDC_cont_all[:N_train,:] # Take training set
        HDC_cont_train_cpy = HDC_cont_train_ * 1
        
        # Apply cyclic accumulation with biases and accumulation speed beta_
        
        # -> INSERT YOUR CODE
        
        # Ternary thresholding with threshold alpha_sp:
            
        # -> INSERT YOUR CODE
            

        Y_train = LABELS[:N_train] - 1
        Y_train = Y_train.astype(int)
        
        # Train the HDC system to find the prototype hypervectors, _q meqns quantized
        centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, HDC_cont_train_cpy, gamma, D_b)
        
        # Do the same encoding steps with the test set
        HDC_cont_test_ = HDC_cont_all[N_train:,:]
        HDC_cont_test_cpy = HDC_cont_test_ * 1
        
        # Apply cyclic accumulation with biases and accumulation speed beta_
        
        # -> INSERT YOUR CODE
        
        # Ternary thresholding with threshold alpha_sp:
            
        # -> INSERT YOUR CODE
        
        Y_test = LABELS[N_train:] - 1
        Y_test = Y_test.astype(int)
        
        # Compute accuracy and sparsity of the test set w.r.t the HDC prototypes
        Acc = compute_accuracy(HDC_cont_test_cpy, Y_test, centroids_q, biases_q)
        sparsity_HDC_centroid = np.array(centroids_q).flatten() 
        nbr_zero = np.sum((sparsity_HDC_centroid == 0).astype(int))
        SPH = nbr_zero/(sparsity_HDC_centroid.shape[0])
        local_avg[trial_] = lambda_1 * Acc + lambda_2 * SPH #Cost F(x) is defined as 1 - this quantity
        local_avgre[trial_] = Acc
        local_sparse[trial_] = SPH
        
    return local_avg, local_avgre, local_sparse