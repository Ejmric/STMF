import numpy as np
import numpy.ma as ma
from utils import max_plus, min_plus, get_coordinates
import time
import copy

np.random.seed(0)

class STMF:
    """
    Fit a sparse tropical matrix factorization model for a matrix X.
    such that
        A = U V + E
    where
        A is of shape (m, n)    - data matrix
        U is of shape (m, rank) - approximated row space
        V is of shape (rank, n) - approximated column space
        E is of shape (m, n)    - residual (error) matrix
    """

    def __init__(self, rank=5, criterion='iterations', max_iter=100, initialization='random_vcol',
                 epsilon=0.00000000000001, random_acol_param=5):
        """
        :param rank: Rank of the matrices of the model.
        :param max_iter: Maximum nuber of iterations.
        """
        self.rank = rank
        self.max_iter = max_iter
        self.initialization = initialization
        self.epsilon = epsilon
        self.random_acol_param = random_acol_param
        self.criterion = criterion

    def b_norm(self, A):
        return np.sum(np.abs(A))

    def initialize_U(self, A, m):
        U_initial = np.zeros((m, self.rank))
        k = self.random_acol_param  # number of columns to average
        if self.initialization == 'random':
            low = A.min()
            high = A.max()
            U_initial = low + (high - low) * np.random.rand(m, self.rank)  # uniform distribution
        elif self.initialization == 'random_vcol':
            # each column in U is element-wise average(mean) of a random subset of columns in A
            for s in range(self.rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].mean(axis=1)
        elif self.initialization == 'col_min':
            for s in range(self.rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].min(axis=1)
        elif self.initialization == 'col_max':
            for s in range(self.rank):
                U_initial[:, s] = A[:, np.random.randint(low=0, high=A.shape[1], size=k)].max(axis=1)
        elif self.initialization == 'scaled':
            low = np.min(A)  # c
            high = 0
            U_initial = low + (high - low) * np.random.rand(m, self.rank)
        return ma.masked_array(U_initial, mask=np.zeros((m, self.rank)))

    def assign_values(self, U, V, f, iterations, columns_perm, time):
        columns_perm_inverse = np.argsort(columns_perm)
        self.U = U
        self.V = V[:, columns_perm_inverse]
        self.error = f
        self.iterations = iterations
        self.time = time

    def fit(self, A):
        """
        Fit model parameters U, V.
        :param X:
            Data matrix of shape (m, n)
            Unknown values are assumed to take the value of zero (0).
        """
        start_time = time.time()
        # permute matrix A, columns minimum increasing
        columns_perm = np.argsort(np.min(A, axis = 0))
        A = A[:, columns_perm]
        
        m = A.shape[0]  # number of rows
        n = A.shape[1]  # number of columns

        # initialization of U matrix
        U_initial = self.initialize_U(A, m)
        V = min_plus(ma.transpose(np.negative(U_initial)), A)
        U = min_plus(A, ma.transpose(np.negative(V)))
        D = np.subtract(A, max_plus(U, V))

        # initialization of f values needed for convergence test
        f_old = 1
        f_new = self.b_norm(D)
        f = f_new

        i_list, j_list, k_list = range(m), range(n), range(self.rank)
        U_new, V_new = U, V
        comb = get_coordinates(A)
        iterations = 0

        while abs(f_old - f_new) > self.epsilon:
            f = f_new
            for x in comb:
                i = x[0]
                j = x[1]
                temp = False
                temp_1 = False
                for k in k_list:
                    iterations += 1
                    U_new = copy.deepcopy(U)
                    U_new[i, k] = A[i, j] - V[k, j]
                    V_new = min_plus(np.transpose(np.negative(U_new)), A)
                    U_new = min_plus(A, np.transpose(np.negative(V_new)))
                    f_new = self.b_norm(np.subtract(A, max_plus(U_new, V_new)))
                    if self.criterion == 'iterations' and iterations >= self.max_iter:
                        self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - start_time, 3))
                        return
                    if f_new < f:
                        temp = True
                        break
                if temp:
                    break  # only executed if the inner loop DID break
                for k in k_list:
                    iterations += 1
                    V_new = copy.deepcopy(V)
                    V_new[k, j] = A[i, j] - U[i, k]
                    U_new = min_plus(A, np.transpose(np.negative(V_new)))
                    V_new = min_plus(np.transpose(np.negative(U_new)), A)
                    f_new = self.b_norm(np.subtract(A, max_plus(U_new, V_new)))
                    if self.criterion == 'iterations' and iterations >= self.max_iter:
                        self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - start_time, 3))
                        return
                    if f_new < f:
                        temp_1 = True
                        break
                if temp_1:
                    break  # only executed if the inner loop DID break
            if f_new < f:
                U = U_new
                V = V_new
                f_old = f
                f = f_new
                if self.criterion == 'iterations' and iterations >= self.max_iter:
                    self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - start_time, 3))
                    return
            else:
                print("no solution found!")
                self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - start_time, 3))
                return
        self.assign_values(U, V, f, iterations, columns_perm, round(time.time() - start_time, 3))


    def predict_all(self):
        """
        Return approximated matrix for all
        columns and rows.
        """
        return max_plus(self.U, self.V)
