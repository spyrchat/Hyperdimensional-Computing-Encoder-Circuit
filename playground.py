# Here the unit tests are performed
import numpy as np
import sys
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
print(counter)
print("hello")
