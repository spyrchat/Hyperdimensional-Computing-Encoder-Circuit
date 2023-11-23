"""
Design of a Hyperdimensional Computing Circuit for Bio-signal Classification via Nelder-Mead optimization
and LS-SVM Training.

*MAIN FILE*

Computer-Aided IC Design (B-KUL-H05D7A)

ir. Ali Safa, ir. Sergio Massaioli, Prof. Georges Gielen (MICAS-IMEC-KU Leuven)

(Author: A. Safa)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from HDC_library import lookup_generate, encode_HDC_RFF, evaluate_F_of_x
plt.close('all')

"""
1) HDC_RFF parameters: DO NOT TOUCH
"""
##################################   
#Replace the path "WISCONSIN/data.csv" with wathever path you have. Note, on Windows, you must put the "r" in r'C:etc..'
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
2) Load dataset: if it fails, replace the path "WISCONSIN/data.csv" with wathever 
path you have. Note, on Windows, you must put the "r" in r'C:etc..'
"""
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

"""
3) Generate HDC LUTs and bundle dataset
"""
grayscale_table = lookup_generate(D_HDC, maxval, mode = 1) #Input encoding LUT
position_table = lookup_generate(D_HDC, imgsize_vector, mode = 0) #weight for XOR-ing
HDC_cont_all = np.zeros((X.shape[0], D_HDC)) #Will contain all "bundled" HDC vectors
bias_ = # -> INSERT YOUR CODE #generate the random biases once

for i in range(X.shape[0]):
    if i%100 == 0:
        print(str(i) + "/" + str(X.shape[0]))
    HDC_cont_all[i,:] = encode_HDC_RFF(np.round((maxval - 1) * X[i,:]).astype(int), position_table, grayscale_table, D_HDC)
 
print("HDC bundling finished...")

"""
4) Nelder-Mead circuit optimization and HDC training
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
    
    #Transform lists to numpy array:
    F_of_x = np.array(F_of_x) 
    Accs = np.array(Accs)
    Sparsities = np.array(Sparsities)
    Simplex = np.array(Simplex)
    
    objective_ = [] #Will contain the objective value F(x) for each simplex as the Nelder-Mead search goes on
    STD_ = [] #Will contain the standard deviation of all F(x) as the Nelder-Mead search goes on
    
    # For the details about the Nelder-Mead step, please refer to the course notes / reference, we are simply implementing that
    for iter_ in range(NM_iter):

        STD_.append(np.std(F_of_x))
        if np.std(F_of_x) < STD_EPS and 100 < iter_:
            break #Early-stopping criteria
        
        #1) sort Accs, Sparsities, F_of_x, Simplex, add best objective to array "objective_"
        
        # -> INSERT YOUR CODE
        
        #2) average simplex x_0 
        
        # -> INSERT YOUR CODE
        
        #3) Reflexion x_r
        
        # -> INSERT YOUR CODE
        
        #Evaluate cost of reflected point x_r
        
        # -> INSERT YOUR CODE
        
        if # -> INSERT YOUR CODE:
            F_of_x[-1] = F_curr
            Simplex[-1,:] = x_r
            Accs[-1] = acc_curr
            Sparsities[-1] = sparse_curr
            rest = False
        else:
            rest = True
            
        if rest == True:
            #4) Expansion x_e
            if # -> INSERT YOUR CODE:
                
                # -> INSERT YOUR CODE
                
                #Evaluate cost of reflected point x_e
                
                # -> INSERT YOUR CODE
                
                if # -> INSERT YOUR CODE:
                    F_of_x[-1] = F_exp
                    Simplex[-1,:] = x_e
                    Accs[-1] = acc_exp
                    Sparsities[-1] = sparse_exp
                else:
                    F_of_x[-1] = F_curr
                    Simplex[-1,:] = x_r
                    Accs[-1] = acc_curr
                    Sparsities[-1] = sparse_curr
       
            else:
                #4) Contraction x_c
                if # -> INSERT YOUR CODE:
                    # -> INSERT YOUR CODE:
                elif # -> INSERT YOUR CODE::
                    # -> INSERT YOUR CODE:
                 
                #Evaluate cost of contracted point x_e
                
                # -> INSERT YOUR CODE:
                
                if # -> INSERT YOUR CODE:
                    F_of_x[-1] = F_c
                    Simplex[-1,:] = x_c
                    Accs[-1] = acc_c
                    Sparsities[-1] = sparse_c
                else:
                    #4) Shrinking
                    for rep in range(1, Simplex.shape[0]):
                        # -> INSERT YOUR CODE:
        
    
    ################################## 
    #At the end of the Nelder-Mead search and training, save Accuracy and Sparsity of the best cost F(x) into the ACCS and SPARSES arrays
    idx = np.argsort(F_of_x)
    F_of_x = F_of_x[idx]
    Accs = Accs[idx]
    Sparsities = Sparsities[idx]
    Simplex = Simplex[idx, :]    
    ACCS[optimalpoint] = Accs[0]
    SPARSES[optimalpoint] = Sparsities[0]
    ################################## 


"""
Plot results (DO NOT TOUCH CODE)
Your code above should return:
    SPARSES: array with sparsity of each chosen lambda_
    ACCS: array of accuracy of each chosen lambda_
    objective_: array of the evolution of the Nelder-Mead objective of the last lambda_ under test
    STD_: array of the standard deviation of the simplex of the last lambda_ under test
    
"""
#Plot tradeoff curve between Accuracy and Sparsity
SPARSES_ = SPARSES[SPARSES > 0] 
ACCS_ = ACCS[SPARSES > 0]
plt.figure(1)
plt.plot(SPARSES_, ACCS_, 'x', markersize = 10)
plt.grid('on')
plt.xlabel("Sparsity")
plt.ylabel("Accuracy")

from sklearn.svm import SVR
y = np.array(ACCS_)
X = np.array(SPARSES_).reshape(-1, 1)
regr = SVR(C=1.0, epsilon = 0.005)
regr.fit(X, y)
X_pred = np.linspace(np.min(SPARSES_), np.max(SPARSES_), 100).reshape(-1, 1)
Y_pred = regr.predict(X_pred)
plt.plot(X_pred, Y_pred, '--')

#Plot the evolution of the Nelder-Mead objective and the standard deviation of the simplex for the last run
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(objective_, '.-')   
plt.title("Objective")
plt.grid("on")
plt.subplot(2,1,2)
plt.plot(STD_, '.-') 
plt.title("Standard deviation") 
plt.grid("on")

plt.figure(3)
plt.plot(lambda_sp, ACCS)

plt.figure(4)
plt.plot(lambda_sp, SPARSES)


