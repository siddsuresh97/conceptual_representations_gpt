import numpy as np
from itertools import combinations
from numpy.linalg import inv, svd
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp


# m = 71 ##num rows
# n = 30 ##num cols


#np.random.seed(1)

#U,S,Vh = svd(np.random.randn(m,n), full_matrices=False)

# evaluate cost function, picking subset "select" from rows of U
# def costfun(U,select):
#     m,n = U.shape
#     return 1/n * inv(U[select,:].T @ U[select,:]).trace()



# Use convex relaxation with rounding
# e.g. as in Joshi-Boyd paper:  doi: 10.1109/TSP.2008.2007095
def BoydHeuristic(U,r,m,n):
    X = cp.Variable((n,n), symmetric=True)
    v = cp.Variable(m)
    Z = cp.bmat( [[X, np.eye(n)], [np.eye(n), U.T @ cp.diag(v) @ U]] )
    constraints = [ Z >> 0, 0 <= v, v <= 1, sum(v) == r ]
    prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
    prob.solve(solver=cp.CVXOPT)
    #prob.solve()
    return v.value


def return_rows(centered_mat ,nrows):
    centered_mat = centered_mat.T.astype('float')
    m = centered_mat.shape[0] ##num rows
    n = centered_mat.shape[1] ##num cols
    U,S,Vh = svd(centered_mat, full_matrices=False)
    v = BoydHeuristic(U,nrows,m=m,n=n)
    srtmp = sorted( enumerate(v), key = lambda x : x[1], reverse=True )
    select_rounding = sorted([ x[0] for x in srtmp[:nrows] ])

    return select_rounding
    

# a = return_rows(np.random.randn(n,m),2)
# print(a)