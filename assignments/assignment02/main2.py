#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt



dummy = 0
def adummy(n):
    return np.zeros(n)

###   Thema: Bayessche Formel

def Bayes_Formel(P_B_A, P_A, P_B):
    P_A_B = (P_B_A*P_A)/P_B
    return P_A_B
    

def totale_Wahrscheinlichkeit(P_B_Ai, P_Ai):
    bufarray = (np.array(P_B_Ai) * np.array(P_Ai).T)

    P_B = np.apply_along_axis(np.sum, bufarray.ndim-1, bufarray)

    try:
        return P_B.item()
    except ValueError:
        return P_B

def Test(Korrektheit, a_priori_Wahrscheinlichkeit):
    P_T_K = Korrektheit
    P_T_nK = 1 - P_T_K

    P_K = a_priori_Wahrscheinlichkeit
    P_nK = 1 - P_K

    P_T = totale_Wahrscheinlichkeit(np.array([P_T_K, P_T_nK]),np.array([P_K, P_nK]))

    P_K_T = Bayes_Formel(P_T_K, P_K, P_T)

    return P_T, P_K_T

def Aufgabe_Bayes():
    """Ausgabe der Werte von Beispiel 2.3.10 und Aufgabe 2.3
    sowie plotten der a-posteriori-Wahrscheinlich gegen die a-priori-Wahrscheinlichkeit."""
    
    # Hier Ausgabe der Werte
    ## Beispiel 2.3.10:
    Korrektheit = 0.98
    a_priori = 0.005
    Test_res = Test(Korrektheit, a_priori)
    print("Beispiel 2.3.10: Totale Wahrscheinlichkeit: {}, a-posteriori-Wahrscheinlichkeit: {}".format(Test_res[0], Test_res[1]))

    ## Aufgabe 2.3
    Korrektheit = 0.98
    a_priori = 0.5
    Test_res = Test(Korrektheit, a_priori)
    print("Aufgabe 2.3 50%: Totale Wahrscheinlichkeit: {}, a-posteriori-Wahrscheinlichkeit: {}".format(Test_res[0], Test_res[1]))
    
    Korrektheit = 0.98
    a_priori = 0.001
    Test_res = Test(Korrektheit, a_priori)
    print("Aufgabe 2.3 0.001%: Totale Wahrscheinlichkeit: {}, a-posteriori-Wahrscheinlichkeit: {}".format(Test_res[0], Test_res[1]))

    
    a_priori_Wahrscheinlichkeiten = np.linspace(0, 1, 1000)
    Test_res = Test(Korrektheit, a_priori_Wahrscheinlichkeiten)
    T = Test_res[0]
    P_K_T = Test_res[1]

    plt.plot(a_priori_Wahrscheinlichkeiten, a_priori_Wahrscheinlichkeiten, label='a priori')
    plt.plot(a_priori_Wahrscheinlichkeiten, T, label='total')
    plt.plot(a_priori_Wahrscheinlichkeiten, P_K_T, label='a posteriori')
    plt.legend()
    plt.show()




###   Thema: Verteilungen

rng = np.random.default_rng()

def normal_10_3(rng, n):
    return rng.normal(10,3,n)

def uniform_100_10(rng, n):
    return rng.uniform(100-10,100+10,n)

def mixed_normal(rng, n, M, S, P):
    """M, S, P sind Listen gleicher Länge von Mittelwerten, Standardabweichungen
    und Wahrscheinlichkeiten. Siehe unten für Argumente und Benutzung von mixed_normal(...)"""
    k = len(M)
    idx = rng.choice(k, size=n, p=P)
    means = np.take(M, idx)
    stds = np.take(S, idx)
    return rng.normal(means, stds, n)

def mixed_normal_2(rng, n):
    data = [(-4, 4), (2, 4), (0.5, 0.5)]
    return mixed_normal(rng, n, *data)

def mixed_normal_3(rng, n):
    data = [(-5, 0, 5), (1, 1, 1), (1/3, 1/3, 1/3)]
    return mixed_normal(rng, n, *data)

def mixed_normal_11(rng, n):
    M = np.linspace(0, 100, 11)
    S = np.linspace(2, 5, 11)
    P = np.array([50, 41, 34, 29, 26, 25, 26, 29, 34, 41, 50]) / 385
    return mixed_normal(rng, n, M, S, P)

distributions = [
    normal_10_3,
    uniform_100_10,
    mixed_normal_2,
    mixed_normal_3,
    mixed_normal_11,
    ]

def Aufgabe_Verteilungen():
    """Plotten von Histogrammen für die Verteilungen."""
    n, bins = 10**6, 100
    for distr in distributions:
        X = distr(rng, n)
        plt.hist(X, bins=bins, density=True, histtype='stepfilled', label=distr.__name__)
        plt.legend()
        plt.show()
        


###   Thema: Gesetz der großen Zahlen (GdgZ)

def rel_variance_of_mean(rng, distr, no_samples, no_runs):
    X = distr(rng, no_samples*no_runs).reshape(no_samples, no_runs)

    # Hier Berechnunge der Mittelwerte über die Spalten von X
    X_mean = np.mean(X,axis= 0)
    # und Berechnung der Varianzen
    X_var = np.var(X_mean)

    X_var_tot = np.var(X.flatten())

    return np.array([X_var/X_var_tot])

def Aufgabe_GdgZ():
    """Hier sollen die relativen Varianzen in Abhängigkeit von der Anzahl der 
    Samples (no_samples) geplottet werden."""
    no_runs = 1000

    No_samples = np.arange(1,101,1)
    
    #Als Plotbefehle könnten Sie folgendes nutzen:
    for distr in distributions:
        Variances = [rel_variance_of_mean(rng,distr,samples, no_runs) for samples in No_samples]
        plt.plot(No_samples, Variances,"o", label=distr.__name__, alpha=0.5)
    plt.legend()
    plt.show()



###   Thema: Zentraler Grenzwertsatz (ZGWS)

def centralized_sample(rng, distr, no_samples, no_runs):
    X = distr(rng, no_samples*no_runs).reshape(no_samples, no_runs)

    #get mean and std of given distribution by using law of large numbers:
    sample = distr(rng,10**4)
    S_mean = np.mean(sample)
    S_std = np.std(sample)


    # Berechnung der zentralisierten Zufallsvariable aus den Spalten von X
    S_sum = np.sum(X, axis = 0)

    Z_n = (S_sum - no_samples *S_mean) / (S_std * np.sqrt(no_samples))


    return Z_n

def Aufgabe_ZGWS():
    no_runs = 10**6
    n = 105
    N = np.arange(100, n+1)
    bins = 100
    for distr in distributions:
        for n in N:
            X = centralized_sample(rng, distr, n, no_runs)
            plt.hist(X, bins=bins, density=True, histtype='stepfilled', alpha=0.6, label=distr.__name__ + f', n={n}');
        plt.legend()
        plt.show()
        

###   Konfiguration von test2.py durch Dictionary 'test_control'

# Folgendees Beispiel schaltet 'test_ZGWS' und 'test_GdgZ' aus
# und testet nur die erste Verteilung 'normal_10_3'.

# test_control = {
#     'test_Verteilungen': [],
#     'test_ZGWS':         [],
#     'test_GdgZ':         [1,2,3,4,5],
#     }


###   Ausführen der Bearbeitungen
if __name__ == '__main__':

    # Aufgabe_Bayes()

    # #Testing:
    # normal_10_3(rng,100)
    # uniform_100_10(rng,100)
    # Aufgabe_Verteilungen()


    # Aufgabe_GdgZ()
    Aufgabe_ZGWS()

