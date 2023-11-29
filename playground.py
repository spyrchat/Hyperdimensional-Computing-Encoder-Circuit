# Here the unit tests are performed
import numpy as np
import sys
from scipy.linalg import lu_factor, lu_solve
np.set_printoptions(threshold=sys.maxsize)

def lookup_generate(dim, n_keys, mode = 1):
    table = np.empty((0, dim)) 
    if mode == 0:
        for i in range(n_keys):    
            row =  np.random.choice([-1, 1], size=(dim), p=[0.5, 0.5])
            table = np.vstack((table, row))
    else:
        for i in range(n_keys):
            probability = i / (n_keys - 1)
            row =  np.random.choice([-1, 1], size=(dim), p=[1-probability, probability])
            table = np.vstack((table, row))

    return table.astype(np.int8)

#print(lookup_generate(5, 256, 1))

xor_result = (lookup_generate(5, 256, 1) ^ lookup_generate(5, 256, 0)).astype(int)
xor_result = (xor_result != 0).astype(int)
xor_result[xor_result == 0] = -1
counter = xor_result.sum(axis=0)

N_train =200
Y_train =  np.random.choice([-1, 1], size=(N_train))
#print(Y_train)
HDC_cont_train = np.random.choice([-1,0,1], size=(N_train,1))
gamma = 2
Beta = np.zeros((N_train+1, N_train+1)) #LS-SVM regression matrix
omega = np.zeros((N_train, N_train))

for i in range(N_train):
    for j in range(N_train):
        omega[i, j] = Y_train[i] * Y_train[j] * HDC_cont_train[i] * HDC_cont_train[j]
        
Beta[1:N_train+1,0] = Y_train
Beta[0,1:N_train+1] = np.transpose(Y_train)
Beta[1:N_train+1,1:N_train+1] = omega + pow(gamma,-1) * np.identity(N_train)


        
#Target vector L:
            
L = np.zeros(N_train+1)
L[1:N_train+1] = np.ones(N_train)
        
#Solve the system of equations to get the vector alpha:
            
v = np.zeros(N_train+1)
lu, piv = lu_factor(Beta)
v = lu_solve((lu,piv),L)
alpha = v[1:N_train+1]
b = v[0]
print(v)
print(len(alpha))     
print(b) 