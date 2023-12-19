from HDC_library import *
import numpy as np
import matplotlib.pyplot as plt

def test(N=1000,B=8,dim=16):
    print("Starting tests for lookup_generate")
    print("\tTest mode 0 started")
    test_mode_0(N,B,dim)
    print("\tTest mode 0 done")

    print("\tTest mode 1 started")
    test_mode_1(N,B,dim)
    print("\tTest mode 1 done")

    print("Starting tests for encode_HDC_RFF")
    print("\t Starting tests for xor")
    test_xor(N)
    print("\t\t All tests passed")

    #print("\t Starting automatic tests")
    #test_auto(N)
    #print("\t\t All tests passed")
    
    print("\t Starting manual tests")
    test_man()
    print("\t\t All tests passed")

def test_mode_1(N,B,dim):
    test_mode_x(N,B,dim,1)

def test_mode_0(N,B,dim):
    test_mode_x(N,B,dim,0)

def test_mode_x(N,B,dim,mode):
    total = N*dim*1.0
    count_pos = np.zeros(2**B)

    for k in range(N):
        result = lookup_generate(dim,2**B,mode)

        assert len(result) == 2**B
        assert len(result[0]) == dim

        for r in range(2**B):
            count_pos[r] += np.sum([k if k == 1 else 0 for k in result[r]])

    plt.figure()
    plt.plot(count_pos/total)
    plt.title("Testing mode {}".format(mode))
    plt.ylim(0,1)
    plt.xlim(0,2**B-1)
    plt.ylabel("Probability [-]")
    plt.xlabel("Row number [-]")
    plt.grid(visible=True)
    if True:
        plt.show(block=False)
        plt.pause(0.01)
    else:
        plt.show()

def forEach(function,arr):
    for el in arr:
        if function(el) == False:
            return False
    return True

def test_xor(N):
    mode=0
    B=8
    dim=100
    for k in range(N):
        W = lookup_generate(dim,2**B,mode)
        L = lookup_generate(dim,2**B,mode)

        WxorL = encode_xor(W,L)

        if (L == -1).all():
            continue
        assert (WxorL != W).any()

        WxorLxorL = encode_xor(WxorL,L)
        assert(WxorLxorL == W).all()
        print("\r\t\t Pass tests for xor: ",k+1, "/",N,end="")
        print("")

def encode_xor(position_table, grayscale_table):
    xor_result = (grayscale_table ^ position_table)
    xor_result = (xor_result != 0).astype(int)
    xor_result[xor_result == 0] = -1
        
    return xor_result

def test_auto(N):
    n_keys=30
    B=8
    dim=2**B
    test_cnt=0
    for test in range(N):
        #W = -np.ones((n_keys,dim))
        W = lookup_generate(dim,n_keys,0)
        L = lookup_generate(dim,2**B,0)
        x = np.random.randint(low=0,high=2**B,size=(n_keys,1),dtype=int)
        encoded = encode_HDC_RFF(x,W,L,dim)
        if (np.linalg.det(L.T) == 0):
            continue

        y = np.around(np.linalg.solve(L.T,encoded)).astype(int)
        x = x.T.tolist()[0]

        for i in range(len(y)):
            for k in range(y[i]):
                assert i in x # if False, error
                x.remove(i)

        assert len(x) == 0
        test_cnt += 1
        print("\r\t\t Pass tests for aut: ", test_cnt, "/", N, end="")
    print("")

def test_man():
    L = np.array([
        [1,-1,-1,-1,-1],
        [1,1,-1,-1,-1],
        [1,1,1,-1,-1],
        [1,1,1,1,-1],
        [1,1,1,1,1]
    ])
    W = np.array([
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1]
    ])

    x = np.array([0,1,2,3,4])
    expected = [-5,-3,-1,1,3]
    result = encode_HDC_RFF(x,W,L,5)
    try:
        assert forEach(lambda x : x == True, [expected[i] == result[i] for i in range(5)])
        print("\t\t Test 1 OK")
    except:
        print("Test 1 Failed")
        print("Got {} but expected {}".format(result,expected))

    x = np.array([4,3,2,1,0])
    expected = [-5,-3,-1,1,3]
    result = encode_HDC_RFF(x,W,L,5)
    try:
        assert forEach(lambda x : x == True, [expected[i] == result[i] for i in range(5)])
        print("\t\t Test 2 OK")
    except:
        print("Test 2 Failed")
        print("Got {} but expected {}".format(result,expected))

test(1000)