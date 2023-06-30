#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
import scipy.stats as st
import warnings

warnings.filterwarnings("ignore")


### Part 1: Maximum likelihood estimation

def L(theta, x, f):
    
    likelihood = np.prod(f(x, theta))
    
    return likelihood


def l(theta, x, f):
    
    log_likelihood = np.sum(np.log(f(x, theta)))
    
    return log_likelihood

def MLE(x, f, theta0, method=None, log=False, return_status=False):
    """"Calculate MLE numerically.
    x iterable: sample,
    f(x_i, theta): densitiy of distribution for x_i with parameter theta,
    theta0: initial guess for finding the maximum; this defines dimension of parameter space."""
    
    
    # Define the negative log-likelihood function for optimization
    if log:
        neg_log_likelihood = lambda theta: -l(theta, x, f)
    else:
        neg_log_likelihood = lambda theta: -L(theta, x, f)
    
    # Use the minimize function for optimization
    result = minimize(neg_log_likelihood, theta0, method=method)
    
    if return_status:
        return result
    else:
        return result.x
    
    
    #return (0, )



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
                    (1.0, 0.1),                                     # theta for generation  (mu, sigma)
                    lambda theta, n: rng.normal(theta[0], theta[1], n),  # generator n samples with parameters mu, sigma
                    lambda x, theta: st.norm.pdf(x, loc=theta[0], scale=theta[1]),   # density function
                    lambda x: [np.mean(x), np.sqrt((1/len(x))*np.sum((x-np.mean(x))**2))],              # calculated exact MLE (mu, sigma=np.std(x))  
                    (0, 1),                                         # initial guess theta0
                    ),    
             'exp': (                                               # exponential distribution
                    2.0, 

                    # Exponentialverteilung mit Parameter lmbda                                           # theta for generation
                    lambda lmbda, n: rng.exponential(scale=1/lmbda, size=n),  # generator n samples with parameter lambda
                    lambda x, lmbda: st.expon.pdf(x, scale=1/lmbda),  # density function
                    lambda x: 1 / np.mean(x),                       # calculated exact MLE (lambda)
                    1,                                              # initial guess theta0
                    ),    
             'uniform': (                                           # uniform distribution
                   (2.0, 5.0),                                      # theta for generation
                   
                   # Gleichverteilung auf dem Intervall [a, b] mit Parameter theta = (a, b)       
                   lambda theta, n: rng.uniform(theta[0], theta[1], n),            # generator n samples with parameters a, b
                   lambda x, theta: st.uniform.pdf(x, loc=theta[0], scale=theta[1]-theta[0]),  # density function
                   lambda x: [np.min(x), np.max(x)],                # calculated exact MLE (a, b)
                   (1, 6),                                          # initial guess theta0
                   ),    
            'binomial10': (                                         # binomial distribution
                   0.3,                                             # theta for generation
                   
                   # Binomialverteilung mit Parameter p und festem Umfang n = 10
                   lambda p, n: rng.binomial(n=10, p=p, size=n),     # generator n samples with parameters p, n=10
                   lambda x, p: st.binom.pmf(x, n=10, p=p),          # probability mass function
                   lambda x: np.sum(x) / 10,                        # calculated exact MLE (p)
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
    # (Intervallschätzung von mu bei bekanntem sigma, Input Samplegröße n und Mittelwert mean)
    
    z = st.norm.ppf(1 - alpha / 2)  # Compute the z-score for the given confidence level
    
    left = mean - z * (sigma / np.sqrt(n))  # Compute the left interval boundary
    right = mean + z * (sigma / np.sqrt(n))  # Compute the right interval boundary
    
    return left, right


def interval_normal_mu_sigma(alpha, n, mean, std):
    # (Intervallschätzung von mu und sigma, Input Samplegröße n, Mittelwert mean und Standardabweichung std)
    
    t = st.t.ppf(1 - alpha / 2, df=n - 1)  # Compute the t-score for the given confidence level and degrees of freedom
    
    left_mu = mean - t * (std / np.sqrt(n))  # Compute the left boundary for mu
    right_mu = mean + t * (std / np.sqrt(n))  # Compute the right boundary for mu
    
    left_sigma = std * np.sqrt((n - 1) / st.chi2.ppf(1 - alpha / 2, df=n - 1))  # Compute the left boundary for sigma
    right_sigma = std * np.sqrt((n - 1) / st.chi2.ppf(alpha / 2, df=n - 1))  # Compute the right boundary for sigma
    
    return (left_mu, right_mu), (left_sigma, right_sigma)


def interval_normal_mu_sample(alpha, sigma, x):
    # (Intervallschätzung von mu bei bekanntem sigma, Input Sample x)
    
    n = len(x)  # Sample size
    mean = np.mean(x)  # Sample mean

    
    
    return interval_normal_mu(alpha, sigma, n, mean)



def interval_normal_mu_sigma_sample(alpha, x):
    # interval_normal_mu_sigma_sample(alpha, x)
    
    
    n = len(x)  # Sample size
    mean = np.mean(x)  # Sample mean
    std = np.std(x, ddof=1)  # Sample standard deviation (unbiased estimator)
    
    
    return interval_normal_mu_sigma(alpha, n, mean, std)



if __name__ == "__main__":
    test_MLE_precision(np.random.default_rng(171717))



