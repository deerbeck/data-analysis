#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
import scipy.stats as st
import warnings

warnings.filterwarnings("ignore")


### Part 1: Maximum likelihood estimation

def L(theta, x, f):
    return np.prod(f(x,theta))

def l(theta, x, f):
    return np.sum(np.log(f(x,theta)))


def MLE(x, f, theta0, method=None, log=False, return_status=False):
    """"Calculate MLE numerically.
    x iterable: sample,
    f(x_i, theta): densitiy of distribution for x_i with parameter theta,
    theta0: initial guess for finding the maximum; this defines dimension of parameter space."""

    if log:
        objective_func = lambda theta: -l(theta, x, f)
    else:
        objective_func = lambda theta: -L(theta, x, f)
    
    result = minimize(objective_func, theta0, method=method)
    
    if return_status:
        return result
    else:
        return result.x


def define_model_test_data(rng):
    return  {
             'normal_std1': (                                       # normal distribution with known sigma=1 and parameter mu
                    10,                                             # theta = mu for generation
                    lambda mu, n: rng.normal(mu, 1, n),             # generator n samples with parameter mu
                    lambda x, mu: st.norm.pdf(x, loc=mu, scale=1),  # densitiy function
                    lambda x: np.mean(x),                           # calculated exact MLE
                    1,                                              # initial guess theta0
                    ),
             'normal': (                                            # normal distribution
                    (1.0, 0.1),                                     # theta for generation
                    lambda theta, n: rng.normal(theta[0], theta[1], n),
                    lambda x, theta: st.norm.pdf(x,loc=theta[0],scale=theta[1]),
                    lambda x: (np.mean(x), np.sqrt((1/len(x))*np.sum((x-np.mean(x))**2))),
                    (0, 1),                                         # initial guess theta0
                    ),    
             'exp': (                                               # exponential distribution
                    2.0,                                            # theta for generation
                    lambda lmbda, n: rng.exponential(scale = 1/lmbda, size = n),
                    lambda x, lmbda: st.expon.pdf(x, scale = (1/lmbda)),
                    lambda x: 1/np.mean(x),
                    1,                                              # initial guess theta0
                    ),
             'uniform': (                                           # uniform distribution
                   (2.0, 5.0),                                      # theta for generation
                   lambda theta, n: rng.uniform(theta[0],theta[1],n),
                   lambda x, theta: st.uniform.pdf(x, loc = theta[0], scale = (theta[1]-theta[0])),
                   lambda x: (min(x),max(x)),
                   (1, 6),                                          # initial guess theta0
                   ),    
            'binomial10': (                                         # binomial distribution
                   0.3,                                             # theta for generation
                   lambda p, n: rng.binomial(10, p, n),
                   lambda x, p: st.binom.pmf(x, p=p, n=10),
                   lambda x: np.mean(x)/10,
                   0.5,                                             # initial guess theta0
                   ),    
            }

def test_MLE_precision(rng, verbose=True):
    test_data = define_model_test_data(rng)
    N = 100
    counts = {}
    for name, data in test_data.items():
        theta, generator, density, MLE_exact, theta0 = data
        for method in None, 'Nelder-Mead', 'SLSQP':   # test these optimization algorithms
            for log in False, True:                   # test Likelihood and log-Likelihood
                for n in 10, 100, 1000:               # test these sample sizes
                    out = f'{name:<12} {str(method):<12} {"l" if log else "L"}  {n:>4} :   '
                    count = np.zeros(10, dtype=int)   # count[k] will contain test with error less than 10**-k
                    for _ in range(N):
                        x = generator(theta, n)       # generate sample of size n
                        MLE_value = MLE_exact(x)
                        ret = MLE(x, density, theta0, method=method, log=log, return_status=True)
                        if ret.success:
                            count[0] += 1
                            error = ret.x - MLE_value
                            for k in range(1, 10):
                                if np.all(np.abs(error) <= 10**-k):
                                    count[k] += 1
                    if verbose:
                        print(out, f'{count[0]:>3}  ', ' '.join(map(str, count[1:])))
                    counts[(name, method, log, n)] = count
    return counts


### Part 2: Confidence interval estimation

def interval_normal_mu(alpha, sigma, n, mean):
    # Intervallschätzung von mu bei bekanntem sigma
    z = st.norm.ppf((1 - alpha/2))

    margin_of_error = z * sigma / np.sqrt(n)

    lower_limit = mean - margin_of_error
    upper_limit = mean + margin_of_error

    return lower_limit, upper_limit

def interval_normal_mu_sigma(alpha, n, mean, std):
    # Intervallschätzung von mu und sigma
    t_value = st.t.ppf(1 - alpha/2, df=(n-1))

    margin_of_error_mu = t_value * std / np.sqrt(n)

    lower_limit_mu = mean - margin_of_error_mu
    upper_limit_mu = mean + margin_of_error_mu


    lower_limit_sigma = std * np.sqrt((n-1) /  st.chi2.ppf(1 - alpha/2, df=n-1))
    upper_limit_sigma = std * np.sqrt((n-1) /  st.chi2.ppf(alpha/2, df=n-1))

    return (lower_limit_mu, upper_limit_mu), (lower_limit_sigma, upper_limit_sigma)

def interval_normal_mu_sample(alpha, sigma, x):
    # Intervallschätzung von mu bei bekanntem sigma
    n = len(x)
    mean = np.mean(x)
    return interval_normal_mu(alpha, sigma, n, mean)

def interval_normal_mu_sigma_sample(alpha, x):
    # Intervallschätzung von mu und sigma
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    return interval_normal_mu_sigma(alpha, n, mean, std)



if __name__ == "__main__":
    test_MLE_precision(np.random.default_rng(171717))
    

    # test_data = define_model_test_data(np.random.default_rng(171717))
    # N = 100
    # n = 100
    # name = "normal"
    # log = False
    # method = "Nelder-Mead"
    # theta, generator, density, MLE_exact, theta0 = test_data["normal"]
    # for n in 10, 100:               # test these sample sizes
    #                 out = f'{name:<12} {str(method):<12} {"l" if log else "L"}  {n:>4} :   '
    #                 count = np.zeros(10, dtype=int)   # count[k] will contain test with error less than 10**-k
    #                 for _ in range(N):
    #                     x = generator(theta, n)       # generate sample of size n
    #                     MLE_value = MLE_exact(x)
    #                     ret = MLE(x, density, theta0, method=method, log=log, return_status=True)
    #                     if ret.success:
    #                         print(ret.x)
    #                         count[0] += 1
    #                         error = ret.x - MLE_value
    #                         for k in range(1, 10):
    #                             if np.all(np.abs(error) <= 10**-k):
    #                                 count[k] += 1
    # print(out, f'{count[0]:>3}  ', ' '.join(map(str, count[1:])))

    