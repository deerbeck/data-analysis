import numpy as np
import math

# calculate arithmetic mean value of the numbers in x


def mittel(x):
    x = np.array(x)
    return sum(x)/(len(x))


# calculate p-Quartil of numbers in x and the
def quartil(x, p):
    x = np.array(x)
    n = len(x)

    # calculate "fractal index"
    o = p*(n-1) + 1

    # gaussian bracket on o for easier use
    o_floor = math.floor(o)

    # calculate quartil using given formula
    return (1-(o-o_floor))*x[o_floor-1] + (o-o_floor)*x[o_floor]


def median(x):
    # considering DRY you can calculate the median as a quartil with p = 0.5
    # return quartil(x, 0.5)

    # calculating median through definition by cases
    x = np.array(x)
    n = len(x)

    #ternary operater for 1 line statement
    return 0.5*(x[int(n/2)-1] + x[int(n/2)]) if n % 2 == 0 else x[int((n+1)/2)-1]


def var(x):
    pass


def regres(x, y):
    pass


def pca(X):
    pass


x = [4.1, 4.5, 4.9, 5.1, 5.4, 5.7, 5.8, 5.8, 6, 67]

print(quartil(x, 0.25))
print(median(x))
