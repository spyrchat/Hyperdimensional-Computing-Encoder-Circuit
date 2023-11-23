import matplotlib.pyplot as plt
import numpy as np

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


LUT,p_in = lookup_generate(1024,256,1)
test_matrix_probability(LUT,p_in)



