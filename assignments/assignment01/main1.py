import numpy as np
import math
from sklearn.decomposition import PCA

np.random.seed(171717)
x4 = np.random.randn(10000)
# assignment 01

# calculate arithmetic mean value of the numbers in x
def mittel(x):
    x = np.array(x)
    return np.sum(x)/len(x)


# calculate p-Quartil of numbers in x with given p
def quantil(x, p):
    x = np.array(x)
    x.sort()
    n = len(x)

    # calculate "fractal index"
    o = p*(n-1)+1

    # gaussian bracket on o for easier use
    o_floor = math.floor(o)

    d = o - o_floor
    
    #distinguish different cases
    if n == 1 and  0<= p <= 1:
        x_p = x[0]

    elif n != 1 and p == 1:
        x_p = x[-1] 

    else:
    # calculate quartil using given formula
        x_p = (1-(d))*x[o_floor-1] + (d)*x[o_floor]
    return x_p

# calculate median of numbers in x
def median(x):
    x = np.array(x)
    x.sort()
    n = len(x)

    # calculating median through definition by cases
    # ternary operater for 1 line statement
    return 0.5*(x[int(n/2)-1] + x[int(n/2)]) if n % 2 == 0 else x[int((n+1)/2)-1]


# calculate uncorrected samplevariance
def var(x):
    x = np.array(x)
    n = len(x)

    # get x_mean needed for s_sqr calculation
    x_mean = mittel(x)
    # get uncorrected samplevariance using given formula
    s_sqr = np.sum((x-x_mean)**2)/n
    return s_sqr


def regress(x, y):

    x = np.array(x)
    y = np.array(y)
    n = len(x)

    #get mean values of x and y
    x_mean = mittel(x)
    y_mean = mittel(y)

    #get empiric covariance
    s_xy = np.sum((x-x_mean)*(y-y_mean))/n

    #calculate slope beta
    beta = float(s_xy/var(x))

    #calculate y-intercept
    alpha = float(y_mean - beta*x_mean)

    #calculate quadratic error
    q = np.sum((y - (alpha + x*beta))**2)
    
    return (beta, alpha, q)


def standardize_sign(Q):
     Q = Q.copy()   # don't change Q
     for col in Q.T:
         for q in col:
             if q > 0:
                 break
             if q < 0:
                 col *= -1
                 break
     return Q

def pca(X):
    n = len(X)

    #get mean array of X
    X_mean = X.mean(axis = 0)

    #get centered sample matrix
    B = X-X_mean

    #get sample covariance matrix
    C = (B.T @ B)/(n-1)

    #eigenvalues 
    e_values, Q = np.linalg.eig(C)

    #sort in descending order
    idx = e_values.argsort()[::-1]
    e_values = e_values[idx]
    Q = Q[:, idx]

    Q = standardize_sign(Q)
    return (Q, e_values, (B @ Q))




# x = [4.1, 4.5, 4.9, 5.1, 5.4, 5.7, 5.8, 5.8, 6, 67]
# x = [90, 100, 110]
# print(quartil(x, 0.25))
# print(median(x))
# print(var(x))

# X = np.array([np.array([1,2]),np.array([2,1]),np.array([3,3]),np.array([4,5]),np.array([5,4])])
# print(pca(x4))