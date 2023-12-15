import numpy as np
from sklearn.utils import shuffle
from scipy.linalg import lu_factor, lu_solve

# Receives the HDC encoded test set "HDC_cont_test" and test labels "Y_test"
# Computes test accuracy w.r.t. the HDC prototypes (centroids) and the biases found at training time

def compute_accuracy(HDC_cont_test, Y_test, centroids, biases):
    Acc = 0
    n_class = int(np.max(Y_test) + 1)
    for i in range(Y_test.shape[0]): # Y_test.shape[0] = rows of Y_test = each patient
        received_HDC_vector = (HDC_cont_test[i])
        all_resp = np.zeros(n_class)
        for cl in range(n_class): # classes is true or false (cancer or no cancer)
            final_HDC_centroid = (centroids[cl])
            #compute LS-SVM response
            response = np.dot(np.transpose(final_HDC_centroid),received_HDC_vector) + biases[cl]

            all_resp[cl] = response
        class_idx = np.argmax(all_resp)
        
        if class_idx == 0:
            class_idx = 1
        else:
            class_idx = -1
        
        if class_idx == Y_test[i]:
            Acc += 1      
    return Acc/Y_test.shape[0]


# Generates random binary matrices of -1 and +1
# when mode == 0, all elements are generated with the same statistics (probability of having +1 always = 0.5)
# when mode == 1, all the probability of having +1 scales with an input "key" i.e., when the inputs to the HDC encoded are coded
# on e.g., 8-bit, we have 256 possible keys
def lookup_generate(dim, n_keys, mode = 1):
    table = np.zeros((n_keys, dim)) 
    prob_array = [0] * n_keys
    if mode == 0:   
        table =  np.random.choice([-1, 1], size=(n_keys,dim), p=[0.5, 0.5])
    else:
        for i in range(n_keys):
            probability = i / (n_keys-1)
            row =  np.random.choice([-1, 1], size=(dim), p=[1-probability, probability])
            table[i,:] = row
            prob_array[i] = probability

    return table.astype(np.int8)#,prob_array

# dim is the HDC dimensionality D
def encode_HDC_RFF(img, position_table, grayscale_table, dim):
    #img contains the 30 features of the current patient
    img_hv = np.zeros(dim, dtype=np.int16)
    container = np.zeros((len(position_table), dim))

    #Get the input-encoding and XOR-ing result: 
    encoded_input = grayscale_table[img] #select the rows from grayscale_table that correspond to the elements of img
    for pixel in range(len(position_table)):
        xor_result = (encoded_input[pixel] ^ position_table[pixel])
        xor_result = (xor_result != 0).astype(int)
        xor_result[xor_result == 0] = -1

        hv = xor_result
        container[pixel, :] = hv*1
        
    img_hv = np.sum(container, axis = 0) #bundling without the cyclic step yet
    return img_hv #,container


# Train the HDC circuit on the training set : (Y_train, HDC_cont_train)
# n_class: number of classes
# N_train: number of data points in training set
# gamma: LS-SVM regularization
# D_b: number of bit for HDC prototype quantization-2
def train_HDC_RFF(n_class, N_train, Y_train_init, HDC_cont_train, gamma, D_b):
    centroids = []
    centroids_q = []
    biases_q = []
    biases = []
    Y_train = np.array(Y_train_init)
    for cla in range(n_class):
        if cla == 1:
            Y_train = Y_train*(-1)
        #The steps below implement the LS-SVM training, check out the course notes, we are just implementing that
        #Beta.alpha = L -> alpha (that we want) 
        Beta = np.zeros((N_train+1, N_train+1)) #LS-SVM regression matrix
        omega = np.zeros((N_train, N_train))
        #Fill Beta:
        for i in range(N_train):
            for j in range(N_train):
                omega[i, j] = Y_train[i] * Y_train[j] * np.dot(np.transpose(HDC_cont_train[i]), HDC_cont_train[j])
        
        Beta[1:N_train+1,0] = Y_train
        Beta[0,1:N_train+1] = np.transpose(Y_train)
        Beta[1:N_train+1,1:N_train+1] = omega + pow(gamma,-1) * np.identity(N_train)
        
        #Target vector L:
        L = np.zeros(N_train+1)
        L[1:N_train+1] = np.ones(N_train)
        
        #Solve the system of equations to get the vector alpha:     
        alpha = np.zeros(N_train+1)
        alpha = np.linalg.solve(Beta,L) #alpha here is the whole v vector from the slides
        #print("beta =",Beta)
        #print("L =",L)
        #print("alpha =",alpha)

        # Get HDC prototype for class cla, still in floating point
        final_HDC_centroid = np.zeros(100)
        final_HDC_centroid_q = np.zeros(100)

        for i in range(N_train):
            final_HDC_centroid = final_HDC_centroid + Y_train[i]*alpha[i+1]*HDC_cont_train[i] #this is mu(vector) from the slides
        
        # Quantization
        max_centroid = np.max(np.abs(final_HDC_centroid))
        final_HDC_centroid_q = np.round(final_HDC_centroid*(2**(D_b-1)-1)/max_centroid)

        #Amplification factor for the LS-SVM bias
        fact = (2**(D_b-1)-1)/max_centroid

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
        bias_train = bias_[:N_train]
        # Apply cyclic accumulation with biases and accumulation speed beta_
        HDC_cont_train_cpy = HDC_cont_train_cpy*beta_ + bias_train
        cyclic_accumulation_train = HDC_cont_train_cpy % (2 ** B_cnt)
       # Ternary thresholding with threshold alpha_sp:
        for row in range(cyclic_accumulation_train.shape[0]):
            for col in range(cyclic_accumulation_train.shape[1]):
                if cyclic_accumulation_train[row,col] - pow(2,B_cnt-1) > alpha_sp:
                    cyclic_accumulation_train[row,col]  = 1
                elif cyclic_accumulation_train[row,col] - pow(2,B_cnt-1) < -alpha_sp:
                    cyclic_accumulation_train[row,col] = -1
                elif abs(cyclic_accumulation_train[row,col] - pow(2,B_cnt-1)) <= alpha_sp:
                    cyclic_accumulation_train[row,col] = 0

        Y_train = (LABELS[:N_train] - 1)*2-1
        Y_train = Y_train.astype(int)
        
        # Train the HDC system to find the prototype hypervectors, _q meqns quantized
        centroids, biases, centroids_q, biases_q = train_HDC_RFF(n_class, N_train, Y_train, cyclic_accumulation_train, gamma, D_b)
        
        # Do the same encoding steps with the test set
        # put testset equal to training set for unit test 2, we want 100% accuracy
        HDC_cont_test_ = HDC_cont_all[N_train:,:]
        #HDC_cont_test_ = HDC_cont_train_*1
        HDC_cont_test_cpy = HDC_cont_test_ * 1
        bias_test = bias_[N_train:]
        #bias_test = bias_train
        # Apply cyclic accumulation with biases and accumulation speed beta_
        HDC_cont_test_cpy = HDC_cont_test_cpy*beta_ + bias_test
        cyclic_accumulation_test = HDC_cont_test_cpy % (2 ** B_cnt)
        # Ternary thresholding with threshold alpha_sp:
        for row in range(cyclic_accumulation_test.shape[0]):
            for col in range(cyclic_accumulation_test.shape[1]):
                if cyclic_accumulation_test[row,col] - pow(2,B_cnt-1) > alpha_sp:
                    cyclic_accumulation_test[row,col] = 1
                elif cyclic_accumulation_test[row,col] - pow(2,B_cnt-1) < -alpha_sp:
                    cyclic_accumulation_test[row,col] = -1
                elif abs(cyclic_accumulation_test[row,col] - pow(2,B_cnt-1)) <= alpha_sp:
                    cyclic_accumulation_test[row,col] = 0

        Y_test = (LABELS[N_train:] - 1)*2-1
        #Y_test = Y_train*1
        Y_test = Y_test.astype(int)
        
        # Compute accuracy and sparsity of the test set w.r.t the HDC prototypes
        Acc = compute_accuracy(cyclic_accumulation_test, Y_test, centroids_q, biases_q)
        sparsity_HDC_centroid = np.array(centroids_q).flatten() 
        nbr_zero = np.sum((sparsity_HDC_centroid == 0).astype(int))
        SPH = nbr_zero/(sparsity_HDC_centroid.shape[0])
        local_avg[trial_] = lambda_1 * Acc + lambda_2 * SPH #Cost F(x) is defined as 1 - this quantity
        local_avgre[trial_] = Acc
        local_sparse[trial_] = SPH
        
    return local_avg, local_avgre, local_sparsegit config pull.rebase false