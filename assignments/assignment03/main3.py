#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
import scipy.stats as st
import warnings

warnings.filterwarnings("ignore")


### Part 1: Maximum likelihood estimation

def L(theta, x, f):
    return 0

def l(theta, x, f):
    return 0

def MLE(x, f, theta0, method=None, log=False, return_status=False):
    """"Calculate MLE numerically.
    x iterable: sample,
    f(x_i, theta): densitiy of distribution for x_i with parameter theta,
    theta0: initial guess for finding the maximum; this defines dimension of parameter space."""
    return (0, )

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
                    ...
                    (1.0, 0.1),                                     # theta for generation
                    ...
                    (0, 1),                                         # initial guess theta0
                    ),    
             'exp': (                                               # exponential distribution
                    2.0,                                            # theta for generation
                    ...,
                    1,                                              # initial guess theta0
                    ),    
             'uniform': (                                           # uniform distribution
                   (2.0, 5.0),                                      # theta for generation
                   ...
                   (1, 6),                                          # initial guess theta0
                   ),    
            'binomial10': (                                         # binomial distribution
                   0.3,                                             # theta for generation
                   ...,
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
    return 0, 0

def interval_normal_mu_sigma(alpha, n, mean, std):
    return 0, 0

def interval_normal_mu_sample(alpha, sigma, x):
    return 0, 0

def interval_normal_mu_sigma_sample(alpha, x):
    return 0, 0
