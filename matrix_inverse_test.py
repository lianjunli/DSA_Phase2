import numpy as np
from time import clock
from scipy.linalg import eigh as largest_eigh
from scipy.linalg import inv
import matplotlib.pyplot as plt

def two_by_two_BlockInv(A_inv, B, C, D):
    E = A_inv + A_inv.dot(B.dot(inv(D - C.dot(A_inv.dot(B))).dot(C.dot(A_inv))))
    F = -A_inv.dot(B.dot(inv(D - C.dot(A_inv.dot(B)))))
    G = -inv(D-C.dot(A_inv.dot(B))).dot(C.dot(A_inv))
    H = inv(D-C.dot(A_inv.dot(B)))
    row_1 = np.concatenate((E,F),axis=1)
    row_2 = np.concatenate((G, H),axis=1)
    matrix_inv = np.concatenate((row_1, row_2), axis=0)
    return matrix_inv

def sub_matrix_4(H):
    assert H.shape[0] == H.shape[1]
    n = H.shape[0]
    m = n//4
    A_split = np.zeros((4, 4, m, m))
    for i in range(4):
        for j in range(4):
            A_split[i,j,:,:] = H[i*m:(i+1)*m, j*m:(j+1)*m]
    return A_split


np.set_printoptions(suppress=True)

N_set = np.arange(500,5001,500)
round = 3
# N_set = [1000]
inv_time=np.zeros((len(N_set), round))
inv_2x2_time =np.zeros((len(N_set), round))
inv_4x4_time =np.zeros((len(N_set), round))
for i in range(len(N_set)):
    for j in range(round):
        N = N_set[i]
        # k=1000
        # np.random.seed(0)
        X = np.random.random((N,N)) - 0.5
        # X = np.dot(X, X.T) #create a symmetric matrix

        # Whole matrix inverse
        #Scipy
        # start = clock()
        # evals_large, evecs_large = largest_eigh(X, eigvals=(N-k,N-1))
        # elapsed = (clock() - start)
        # print ("eigh elapsed time: ", elapsed)

        start = clock()
        a = inv(X)
        elapsed = (clock() - start)
        print ("inv elapsed time: ", elapsed)
        inv_time[i,j]=elapsed

        #Numpy
        # start = clock()
        # evals_large, evecs_large = np.linalg.eig(X)
        # elapsed = (clock() - start)
        # print ("NP eigh elapsed time: ", elapsed)
        #
        # start = clock()
        # a = np.linalg.inv(X)
        # elapsed = (clock() - start)
        # print ("NP inv elapsed time: ", elapsed)

        #Block matrix inverse 2 by 2
        start = clock()
        X_1 = X[0:N//2, 0:N//2]
        X_2 = X[0:N//2, N//2:N]
        X_3 = X[N//2:N, 0:N//2]
        X_4 = X[N//2:N, N//2:N]
        X_1_inv = inv(X_1)
        block_inv = two_by_two_BlockInv(X_1_inv, X_2,X_3,X_4)
        elapsed = (clock() - start)
        print ("block inv 2x2 elapsed time: ", elapsed)
        inv_2x2_time[i,j]=elapsed

        #Block matrix inverse 4 by 4
        start = clock()
        A_subs = sub_matrix_4(X)
        q0_inv = inv(A_subs[0,0])
        q1_inv = two_by_two_BlockInv(q0_inv, A_subs[0,1],A_subs[1,0],A_subs[1,1])

        q2_inv = two_by_two_BlockInv(q1_inv, np.concatenate((A_subs[0,2],A_subs[1,2]),axis=0), np.concatenate((A_subs[2,0],A_subs[2,1]),axis=1), A_subs[2,2])
        q3_inv = two_by_two_BlockInv(q2_inv, np.concatenate((A_subs[0,3],A_subs[1,3], A_subs[2,3]),axis=0), np.concatenate((A_subs[3,0],A_subs[3,1], A_subs[3,2]),axis=1), A_subs[3,3])
        elapsed = (clock() - start)
        print ("block inv 4x4 elapsed time: ", elapsed)
        inv_4x4_time[i,j]=elapsed

plt.figure()
plt.plot(N_set, np.mean(inv_time,axis=1), 'o-', label='Direct Inv')
plt.plot(N_set, np.mean(inv_2x2_time,axis=1), 'x-', label='2x2 block Inv')
plt.plot(N_set, np.mean(inv_4x4_time,axis=1), '^-', label='4x4 block Inv')
plt.legend(loc=0)
plt.xlabel('Number of clusters')
plt.ylabel('Time (sec)')
plt.title('Matrix Inversion Time')
plt.show()
print('done')



