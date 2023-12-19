from HDC_library import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

def test(N,B,dim):
    print("Starting tests for lookup_generate")
    print("\t Test mode 0")
    test_lookup_generate(N,B,dim,0)

    print("\t Test mode 1")
    test_lookup_generate(N,B,dim,1)

    print("Starting tests for encode_HDC_RFF")
    print("\t Starting tests for XOR")
    test_XOR()
    
    print("\t Starting manual tests")
    test_encode_HDC_RFF()


def test_lookup_generate(N,B,dim,mode):
    total = N*dim
    cntr_plus1 = np.zeros(2**B)

    for i in range(N):
        LUT = lookup_generate(dim,2**B,mode)

        assert len(LUT) == 2**B #rows
        assert len(LUT[0]) == dim #cols

        for r in range(2**B): #for each row
            # calculate the sum of the elements in each row, counting only the elements with a value of 1
            cntr_plus1[r] += np.sum([1 if k == 1 else 0 for k in LUT[r]]) 

    plt.figure()
    plt.plot(cntr_plus1/total)
    plt.title("Testing mode {}".format(mode))
    plt.ylim(0,1)
    plt.xlim(0,2**B-1)
    plt.ylabel("Probability")
    plt.xlabel("Row")
    plt.grid(visible=True)
    plt.savefig("test_possibilites_mode{}.png".format(mode))


def test_XOR():
    mat = spio.loadmat('XOR_test_data.mat', squeeze_me=True)

    in1 = mat['in1']
    in2 = mat['in2']
    desired = mat['result']
    # xor implementation from HDC_library
    calculated = (in1 ^ in2)
    calculated = (calculated != 0).astype(int)

    if (desired == calculated).all():
        print("\t\t XOR result is correct")
    else:
        print("\t\t XOR result is different")

def test_encode_HDC_RFF():
    L = np.array([
        [1,-1,-1,-1,-1],
        [1, 1,-1,-1,-1],
        [1, 1, 1,-1,-1],
        [1, 1, 1, 1,-1],
        [1, 1, 1, 1, 1]
    ])
    W = np.ones((5,5),dtype=np.int8)

    img = np.array([0,1,2,3,4])
    expected = [-5,-3,-1,1,3] # calculated manually
    result = encode_HDC_RFF(img,W,L,5)
    assert all(expected[i] == result[i] for i in range(5)), "Assertion failed: Test 1"
    print("\t\t Test 1 OK")
 
    img = np.array([4,3,2,1,0])
    expected = [-5,-3,-1,1,3]
    result = encode_HDC_RFF(img,W,L,5)
    assert all(expected[i] == result[i] for i in range(5)), "Assertion failed: Test 2"
    print("\t\t Test 2 OK")

test(100,8,100)
