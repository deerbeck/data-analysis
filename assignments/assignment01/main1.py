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
    pass


def var(x):
    pass


def regres(x, y):
    pass


def pca(X):
    pass


x = [4.1, 4.5, 4.9, 5.1, 5.4, 5.7, 5.8, 5.8, 6, 67]

print(quartil(x, 0.25))
