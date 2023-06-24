#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors

def ecdf(X, x):
    n = len(X)  # Gesamtanzahl der Elemente in X
    count = np.sum(X <= x)  # Anzahl der Elemente in X, die kleiner oder gleich x sind
    return count / n  # Verhältnis zur Gesamtanzahl gibt die ECDF an der Stelle x zurück

def plot_empiric_normal(rng):
    n = 10000

    X = rng.normal(0, 1, n)

    x_vals = np.linspace(-5, 5, n)


    # Berechnung der ECDF-Werte
    ecdf_vals = [ecdf(X, x) for x in x_vals]

    # Berechnung der Phi (Standardnormalverteilung)-Werte
    phi_vals = st.norm.cdf(x_vals)

    # Plot der ECDF und Phi
    plt.figure("Verteilungsfuntkionen")
    plt.plot(x_vals, ecdf_vals, label='Empirische Verteilungsfunktion (ECDF)')
    plt.plot(x_vals, phi_vals, label='Verteilungsfunktion Φ (Standardnormalverteilung)')
    plt.xlabel('x')
    plt.ylabel('Wahrscheinlichkeit')
    plt.title('ECDF und Verteilungsfunktion')

    plt.plot(x_vals, (ecdf_vals - phi_vals), label = "Differenz der Wahrscheinlichkeiten")
    plt.legend()
    plt.grid(True)
    plt.show()

alpha = 0.05  

def KS(X, mu, sigma):
    return True

def LF(X):
    return True

def SW(X):
    return True

def AD(X):
    return True

tests = [KS, LF, SW, AD]

rng = np.random.default_rng(171717)

def Normal(mu, sigma):
    def distr(n):
        return rng.normal(mu, sigma, n)
    distr.__name__ = f'normal_{mu}_{sigma}'
    return distr, mu, sigma

def Uniform(a, b):
    def distr(n):
        return rng.uniform(a, b, n)
    distr.__name__ = f'uniform_{a}_{b}'
    return distr, (a + b)/2, np.sqrt((b - a)/12)

def Beta(p, q):
    """Beta-Verteilung"""
    def distr(n):
        return n*[0]
    distr.__name__ = f'beta_{p}_{q}'
    return distr, 0, 0

def T(N):
    """Studentsche t-Verteilung"""
    def distr(n):
        return n*[0]
    distr.__name__ = f't_{N}'
    return distr, 0, 0

def Laplace(mu, sigma):
    """Laplace-Verteilung"""
    def distr(n):
        return n*[0]
    distr.__name__ = f'laplace_{mu}_{sigma}'
    return distr, 0, 0

def Chi2(N):
    """Chi-Quadrat-Verteilung"""
    def distr(n):
        return n*[0]
    distr.__name__ = f'chi2_{N}'
    return distr, 0, 0

def Gamma(a, b):
    """Gamma-Verteilung"""
    k, theta = a, 1/b
    def distr(n):
        return n*[0]
    distr.__name__ = f'gamma_{a}_{b}'
    return distr, 0, 0

distributions = [
                    Normal(3, 9),
                    Uniform(0, 1), 
                    Beta(2, 2),
                    T(300),
                    T(10),
                    T(7),
                    Laplace(0, 1),
                    Chi2(20),
                    Chi2(4),
                    Gamma(4, 5),
                    Gamma(1, 5),
                 ]

# N = 100000   # takes ca. 10h
# N = 10000    # takes ca. 1h
N = 1000       # takes ca. 6min
# N = 100      # takes 40s, not accurate
# N = 10       # takes 5s, not accurate
      
def true_rates(n, distribution):
    """Count success reates."""
    distr, mu, sigma = distribution
    counts = {t.__name__: 0 for t in tests}
    for _ in range(N):
        for test in tests:
            if test == KS:
                counts[test.__name__] += test(distr(n), mu, sigma)
            else:
                counts[test.__name__] += test(distr(n))
    return {name: counts[name] / N for name in counts}
    
def rates_to_str(rates, inv=False):
    """Convert rates to string."""
    return ', '.join([f'{name}: {round(100*(1 - rate if inv else rate)):>3}%' for name, rate in rates.items()])

def test_tests(verbose=True, savefig=False):
    """Run all the normality tests."""
    ni = [10, 20, 30, 50, 100, 200, 300, 400, 500, 1000, 2000]
    # Collect rates
    beta = {}
    for n in ni:
        if verbose:
            print()
        print(f'Testing alpha = {alpha}, N = {N}, n = {n}')
        distribution = Normal(0, 1)
        rate = true_rates(n, distribution)
        if verbose:
            name = distribution[0].__name__
            print(f'Annahme-Rate    für {name + ":":<12} {rates_to_str(rate)}')
        for distribution in distributions:
            name = distribution[0].__name__
            rates = true_rates(n, distribution)
            for test_name, rate in rates.items():
                beta[(test_name, name, n, alpha)] = 1 - rate
            if verbose:
                print(f'Abweisungs-Rate für {name + ":":<12} {rates_to_str(rates, inv=True)}')
    # print and plot results
    for distribution in distributions:
        distr_name = distribution[0].__name__
        for test in tests:
            test_name = test.__name__
            B = [beta[(test_name, distr_name, n, alpha)] for n in ni]
            plt.plot(B, {'SW': 'b', 'AD': 'tab:purple', 'LF': 'g', 'KS': 'r', 'P4': 'tab:orange', 'P5': 'c', 'P6': 'y', 'P7': 'tab:brown'}[test_name], label=test_name)
        ax = plt.gca()
        ax.set_xticks(range(len(ni)))
        ax.set_xticklabels(ni)
        plt.xlabel('Stichprobengröße n')
        if distribution == distributions[0]:
            filename = f'alpha-Fehler_alpha{round(alpha*100)}.pdf'
            plt.title(f'Stichproben-Verteilung {distr_name}, alpha = {alpha}')
            plt.ylabel('alpha-Fehler')
        else:
            filename = f'power_{distr_name}_alpha{round(alpha*100)}.pdf'
            plt.title(f'Stichproben-Verteilung {distr_name}, alpha = {alpha}')
            plt.ylabel('Power = Trennschärfe = 1 - beta')
        plt.legend()
        if savefig:
            plt.savefig(filename)
        plt.show()
        



        
if __name__ == "__main__":
    plot_empiric_normal(rng)
