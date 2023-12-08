import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from HDC_library import train_HDC_RFF
from HDC_library import compute_accuracy
from HDC_library import evaluate_F_of_x
from HDC_library import encode_HDC_RFF
from HDC_library import lookup_generate
from sklearn.utils import shuffle
np.set_printoptions(threshold=np.inf, linewidth=200)
'''
def lookup_generate(dim, n_keys, mode = 1):
    table = np.empty((0, dim)) 
    prob_array = [0] * n_keys
    if mode == 0:
        for i in range(n_keys):    
            row =  np.random.choice([-1, 1], size=(dim), p=[0.5, 0.5])
            table = np.vstack((table, row))
    else:
        for i in range(n_keys):
            probability = i / (n_keys-1)
            row =  np.random.choice([-1, 1], size=(dim), p=[1-probability, probability])
            table = np.vstack((table, row))
            prob_array[i] = probability

    return table.astype(np.int8),prob_array
'''


def test_matrix_probability(LUT,in_p):
    rows = len(LUT)
    print(rows)
    cols = len(LUT[0])
    cntr_plus1 = [0] * rows
    cntr_minus1 = [0] * rows
    p_plus1 = [0] * rows
    for i in range(rows):
        for j in range(cols):
            if LUT[i,j]==1:
                cntr_plus1[i]+=1
            elif LUT[i,j]==-1:
                cntr_minus1[i]+=1
            else:
                print("matrix element is not +1 or -1")
        p_plus1[i] = cntr_plus1[i]/cols
    plt.plot(in_p, p_plus1)
    plt.ylabel("output probability")
    plt.xlabel("input probability")
    plt.show()


#LUT,p_in = lookup_generate(1024,256,1)C
#test_matrix_probability(LUT,p_in)

dim = 1024
#position_table = lookup_generate(dim, len(position_table), 1)
#grayscale_table = lookup_generate(dim, len(position_table), 0)
mat = spio.loadmat('XOR_test_data.mat', squeeze_me=True)

in1 = mat['in1'] # array
in2 = mat['in2']
desired = mat['result']
'''
def encode_HDC_RFF(img, position_table, grayscale_table, dim):
    img_hv = np.zeros(dim, dtype=np.int16)
    container = np.zeros((len(position_table), dim))
    for pixel in range(len(position_table)):
        #Get the input-encoding and XOR-ing result:  
        xor_result = (grayscale_table[pixel] ^ position_table[pixel])
        xor_result = (xor_result != 0).astype(int)

        hv = xor_result
        container[pixel, :] = hv*1
        
    img_hv = np.sum(container, axis = 0) #bundling without the cyclic step yet
    return container
'''

def test_XOR(in1,in2,desired,dim):
    calculated = encode_HDC_RFF([0], in1, in2, dim)
    print("desired =",desired)
    print("calculated =",calculated)
    if (desired == calculated).all():
        print("XOR result is correct")
    else:
        print("XOR result is different")

#test_XOR(in1,in2,desired,dim)


dataset_path = 'WISCONSIN/data.csv' 
##################################   
imgsize_vector = 30 #Each input vector has 30 features
n_class = 2
D_b = 4 #We target 4-bit HDC prototypes
B_cnt = 8
maxval = 256 #The input features will be mapped from 0 to 255 (8-bit)
D_HDC = 100 #HDC hypervector dimension
portion = 0.6 #We choose 60%-40% split between train and test sets
Nbr_of_trials = 1 #Test accuracy averaged over Nbr_of_trials runs
N_tradeof_points = 40 #Number of tradeoff points - use 100 
N_fine = int(N_tradeof_points*0.4) #Number of tradeoff points in the "fine-grain" region - use 30
#Initialize the sparsity-accuracy hyperparameter search
lambda_fine = np.linspace(-0.2, 0.2, N_tradeof_points-N_fine)
lambda_sp = np.concatenate((lambda_fine, np.linspace(-1, -0.2, N_fine//2), np.linspace(0.2, 1, N_fine//2)))
N_tradeof_points = lambda_sp.shape[0]
    
"""
#2) Load dataset: if it fails, replace the path "WISCONSIN/data.csv" with wathever 
#path you have. Note, on Windows, you must put the "r" in r'C:etc..'
"""
DATASET = np.loadtxt(dataset_path, dtype = object, delimiter = ',', skiprows = 1)
X = DATASET[:,2:].astype(float)
LABELS = DATASET[:,1]
LABELS[LABELS == 'M'] = 0
LABELS[LABELS == 'B'] = 2
LABELS = LABELS.astype(float)
X = X.T / np.max(X, axis = 1)
X, LABELS = shuffle(X.T, LABELS)
imgsize_vector = X.shape[1]
N_train = int(X.shape[0]*portion)

"""
#3) Generate HDC LUTs and bundle dataset
"""
grayscale_table = lookup_generate(D_HDC, maxval, mode = 1) #Input encoding LUT
position_table = lookup_generate(D_HDC, imgsize_vector, mode = 0) #weight for XOR-ing
HDC_cont_all = np.zeros((X.shape[0], D_HDC)) #Will contain all "bundled" HDC vectors
bias_ = 0 # -> INSERT YOUR CODE #generate the random biases once

for i in range(X.shape[0]):
    if i%100 == 0:
        print(str(i) + "/" + str(X.shape[0]))
    HDC_cont_all[i,:] = encode_HDC_RFF(np.round((maxval - 1) * X[i,:]).astype(int), position_table, grayscale_table, D_HDC)
 
print("HDC bundling finished...")

"""
#4) Nelder-Mead circuit optimization and HDC training
"""
################################## 
#Nelder-Mead parameters
NM_iter = 350 #Maximum number of iterations
STD_EPS = 0.002 #Threshold for early-stopping on standard deviation of the Simplex
#Contraction, expansion,... coefficients:
alpha_simp = 1 * 0.5
gamma_simp = 2 * 0.6
rho_simp = 0.5
sigma_simp = 0.5
################################## 

ACCS = np.zeros(N_tradeof_points)
SPARSES = np.zeros(N_tradeof_points)
load_simplex = True # Keep it to true in order to have somewhat predictive results
for optimalpoint in range(N_tradeof_points):
    print("Progress: " + str(optimalpoint+1) + "/" + str(N_tradeof_points))
    # F(x) = 1 - (lambda_1 * Accuracy + lambda_2 * Sparsity) : TO BE MINIMIZED by Nelder-Mead
    lambda_1 = 1 # Weight of Accuracy contribution in F(x)
    lambda_2 = lambda_sp[optimalpoint] # Weight of Sparsity contribution in F(x): varies!

    #Initialize the Simplex or either load a pre-defined one (we will use a pre-defined one in the lab for reproducibility)
    if load_simplex == False:
        Simplex = []
        N_p = 11
        for ii in range(N_p):
            alpha_sp = np.random.uniform(0, 1) * ((2**B_cnt) / 2)
            gam_exp = np.random.uniform(-5, -1)
            beta_ = np.random.uniform(0, 2) * (2**B_cnt-1)/imgsize_vector
            gamma = 10**gam_exp
            simp_arr = np.array([gamma, alpha_sp, beta_])
            Simplex.append(simp_arr*1)  
            
        #np.savez("Simplex2.npz", data = Simplex)
             
    else:
        print("Loading simplex")
        Simplex = np.load("Simplex2.npz", allow_pickle = True)['data']

    
    #Compute the cost F(x) associated to each point in the Initial Simplex
    F_of_x = []
    Accs = []
    Sparsities = []
    for init_simp in range(len(Simplex)):
        simp_arr = Simplex[init_simp] #Take Simplex from list
        gamma = simp_arr[0] #Regularization hyperparameter
        alpha_sp = simp_arr[1] #Threshold of accumulators
        beta_ = simp_arr[2] #incrementation step of accumulators
        ############### F(x) for Nelder-Mead with x = [gamma, alpha_sp, beta] ###################
        #The function "evaluate_F_of_x_2" performs:
        #    a) thresholding and encoding of bundled dataset into final HDC "ternary" vectors (-1, 0, +1)
        #    b) Training and testing the HDC system on "Nbr_of_trials" trials (with different random dataset splits)
        #    c) Returns lambda_1*Acc + lambda_2*Sparsity, Accuracy and Sparsity for each trials
        local_avg, local_avgre, local_sparse = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, bias_, gamma, alpha_sp, n_class, N_train, D_b, lambda_1, lambda_2, B_cnt)
        F_of_x.append(1 - np.mean(local_avg)) #Append cost F(x)  
        Accs.append(np.mean(local_avgre))
        Sparsities.append(np.mean(local_sparse))
        ##################################   
