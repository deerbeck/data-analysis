import numpy as np
import math
# assignment 01

# calculate arithmetic mean value of the numbers in x


def mittel(x):
    x = np.array(x)
    return sum(x)/(len(x))


# calculate p-Quartil of numbers in x with given p
def quartil(x, p):
    x = np.array(x)
    n = len(x)

    # calculate "fractal index"
    o = p*(n-1) + 1

    # gaussian bracket on o for easier use
    o_floor = math.floor(o)

    # calculate quartil using given formula
    return (1-(o-o_floor))*x[o_floor-1] + (o-o_floor)*x[o_floor]


# calculate median of numbers in x
def median(x):
    x = np.array(x)
    n = len(x)

    # considering DRY you can calculate the median as a quartil with p = 0.5
    # return quartil(x, 0.5)

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
    s_sqr = (1/n) * sum((x-x_mean)**2)
    return s_sqr


def regres(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    #get mean values of x and y
    x_mean = mittel(x)
    y_mean = mittel(y)

    #get empiric covariance
    s_xy = (1/n)*sum((x-x_mean)*(y-y_mean))

    #get uncorrected samplevariance from x
    s_x_sqr = var(x)

    #calculate slope beta
    beta = (s_xy/s_x_sqr)

    #calculate y-intercept
    alpha = y_mean - beta*x_mean

    #calculate quadratic error
    q = sum(y - ((alpha + beta * x)**2))
    
    return (beta, alpha, q)


def pca(X):
    X_mean = []
    X_T = np.transpose(X)
    n = len(X)

    #get mean array of X
    for i in range(len(X_T)):
        X_mean.append(np.array([mittel(X[i])]*n))
    X_mean = np.array(X_mean).transpose()
    
    X_n = []
    #get centered sample matrix
    B = X-X_mean

    #get sample covariance matrix
    C = []
    C.append((1/(n-1))*np.dot(np.transpose(B),B))
    
    #iterate  throught C to get diagonal matrix as long as cornerpoints are smaller than 1e-15
    while(~np.all(np.abs(C[-1] - np.diag(np.diagonal(C[-1]))) < 1e-15)):
        #QR-decomposition of C to get C_n+1
        Q,R = np.linalg.qr(C[-1])
        C.append(np.dot(R,Q))

    #append best diagonal matrix C to 
    C.append(np.around(C[-1], 10))

    #get diagonal matrix containing eigenvalues
    D = np.dot(np.dot(np.transpose(Q),C),Q)

    pass




# x = [4.1, 4.5, 4.9, 5.1, 5.4, 5.7, 5.8, 5.8, 6, 67]
# x = [90, 100, 110]
# print(quartil(x, 0.25))
# print(median(x))
# print(var(x))

X = np.array([np.array([1,2]),np.array([2,1]),np.array([3,3]),np.array([4,5]),np.array([5,4])])
pca(X)