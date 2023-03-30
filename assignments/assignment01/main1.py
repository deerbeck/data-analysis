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


#calculate uncorrected samplevariance
def var(x):
    x = np.array(x)
    n = len(x)

    #get x_mean needed for s_sqr calculation
    x_mean = mittel(x)
    #get uncorrected samplevariance using given formula
    test = x - x_mean
    test2 = test**2
    s_sqr = (1/n) * sum((x-x_mean)**2)
    return s_sqr


def regres(x, y):
    pass


def pca(X):
    pass


x = [4.1, 4.5, 4.9, 5.1, 5.4, 5.7, 5.8, 5.8, 6, 67]

x = [90,100,110]
print(quartil(x, 0.25))
print(median(x))
print(var(x))