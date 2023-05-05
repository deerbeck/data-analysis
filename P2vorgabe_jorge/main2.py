#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt



dummy = 0
def adummy(n):
    return np.zeros(n)



###   Thema: Bayessche Formel

# Definieren Sie die Formeln wie in der Vorlesung; dabei sind die Argumente floats oder iterables. Test soll ein Paar, also ein 2-
# Tupel, bestehend aus der totalen Wahrscheinlichkeit und der a-posteriori-Wahrscheinlichkeit zurückgeben. Benutzen Sie nun die
# Funktionen, um diese beiden Wahrscheinlichkeiten in Abhängigkeit von der a-priori-Wahrscheinlichkeit zu plotten; siehe main2.py
# für Vorschläge zu den Plot-Befehlen.


# Seite 26 Skript
def Bayes_Formel(P_B_A, P_A, P_B):
    
    P_B_A = float(P_B_A)
    P_A = float(P_A)
    P_B = float(P_B)
    
    
    P_A_B = (P_B_A * P_A)/P_B
    
    return P_A_B


# Seite 26 Skript
def totale_Wahrscheinlichkeit(P_B_Ai, P_Ai):
    
    
    P_B_Ai = list(map(float, P_B_Ai))
    P_Ai = list(map(float, P_Ai))

    P_B = np.sum(np.array(P_B_Ai) * np.array(P_Ai))
    
    return P_B



def Test(Korrektheit, a_priori_Wahrscheinlichkeit):
    
    # a_priori_Wahrscheinlichkeit
    P_K = a_priori_Wahrscheinlichkeit
    
    # Sensitivitaet
    P_T_K = Korrektheit
    
    # totale wahrscheinlichkeit
    P_T  = P_T_K*P_K + (1-P_T_K)*(1-P_K)
    
    # Bayes_Formel: a-posteriori-Wahrscheinlichkeit
    P_K_T = Bayes_Formel(P_T_K, P_K, P_T)
    

    return P_T, P_K_T



def Aufgabe_Bayes():
    """Ausgabe der Werte von Beispiel 2.3.10 und Aufgabe 2.3
    sowie plotten der a-posteriori-Wahrscheinlich gegen die a-priori-Wahrscheinlichkeit."""
    
    # Hier Ausgabe der Werte
    
    # 98%
    Korrektheit = 0.98
    
    # 0.5%, 50%, 0.1%
    a_priori_Wahrscheinlichkeit_en = [0.005, 0.5, 0.001]
    
    print(f'Sensitivitaet {Korrektheit}')
    
    for count, a_priori_Wahrscheinlichkeit in enumerate(a_priori_Wahrscheinlichkeit_en):
        
        P_T, P_K_T = Test(Korrektheit, a_priori_Wahrscheinlichkeit)
        
        
        print('P_K = {}, P_T = {}, P_K_T = {}'.format(a_priori_Wahrscheinlichkeit_en[count], P_T, P_K_T))




    # Die Plotbefehle könnten folgendermaßen aussehen:
    
    # 98%
    Korrektheit = 0.98
    
    # a_priori_Wahrscheinlichkeit_en
    a_priori_Wahrscheinlichkeiten = np.linspace(0, 1, 1000)
    
    list_PT = []
    list_PKT = []
    
    for a_priori_Wahrscheinlichkeit in a_priori_Wahrscheinlichkeiten:
        
        P_T, P_K_T = Test(Korrektheit, a_priori_Wahrscheinlichkeit)
        
        list_PT.append((P_T))
        list_PKT.append((P_K_T))
   
    
    #plt.plot(a_priori_Wahrscheinlichkeiten, a_priori_Wahrscheinlichkeiten, label='a priori')
    
    plt.plot(a_priori_Wahrscheinlichkeiten, list_PT, label='total')
    plt.plot(a_priori_Wahrscheinlichkeiten, list_PKT, label='a posteriori')
    plt.xlabel('a-priori probability P(K)')
    plt.legend()
    plt.show()
    
    
#######################################################################################################
#######################################################################################################

###   Thema: Verteilungen

# Objekt rng der Klasse Generator definiert, mit
# dessen Hilfe sich Zufallszahlen von einer Vielzahl von Verteilungen generieren lassen

rng = np.random.default_rng()

# Arrays der Größe n Zufallszahlen

# (a) der Normalverteilung N(10, 3^2)
def normal_10_3(rng, n):
    
    # M = 10
    # S = 3
    return rng.normal(10, 3, n) # size is n


#  (b) der Gleichverteilung mit Erwartungswert 100 und Standardabweichung 10
def uniform_100_10(rng, n): 
    
    # a + b = 200
    # b = a + 10 * sqrt(12)
    
    a = 100 - 5 * np.sqrt(12)
    b = 100 + 5 * np.sqrt(12)
    
    return rng.uniform(a, b, n) # size is n


#  (c) einer sogenannten gemischten Gaußverteilung
def mixed_normal(rng, n, M, S, P):
    
    
    """M, S, P selbe Laenge k. 
    
    n Samples des Rückgabe-Arrays sollen nach folgender Regel gezogen werden: 
    
    Zunächst wird ein i ∈ {1, 2, . . . , k} mit Wahrscheinlichkeit pi gewählt und dann das Sample gemäß N(µi, σi^2) gezogen
    
    Vielleicht hilft Ihnen die Funktion rng.multinomial aus der Klasses Generator von numpy.random
    
    Siehe unten für Argumente und Benutzung von mixed_normal(...)"""
    
    
    # Mittelwerten/Erwartungs wert: M   
    M = np.array(M)
    
    # Standardabweichungen. S
    S = np.array(S)
    
    # Wahrscheinlichkeiten: P
    P = np.array(P)
    
    # Generate the mixture probabilities for each sample
    mixture_probs = rng.multinomial(1, P, size=n)   # array size n x len(P)
    
    # Generate random samples from each Gaussian distribution
    samples = rng.normal(M, S, size=(n, len(M)))    # array size n x len(M)    M, S, and P same size
   
    # Compute the weighted sum of the samples using the mixture probabilities
    mixed_samples = np.sum(mixture_probs * samples, axis=1)
    
    
    return mixed_samples



def mixed_normal_2(rng, n):
    
    # data --> M, S, P
    data = [(-4, 4), (2, 4), (0.5, 0.5)]
    return mixed_normal(rng, n, *data)

def mixed_normal_3(rng, n):
    
    # data --> M, S, P
    data = [(-5, 0, 5), (1, 1, 1), (1/3, 1/3, 1/3)]
    return mixed_normal(rng, n, *data)

def mixed_normal_11(rng, n):
    M = np.linspace(0, 100, 11)
    S = np.linspace(2, 5, 11)
    P = np.array([50, 41, 34, 29, 26, 25, 26, 29, 34, 41, 50]) / 385
    return mixed_normal(rng, n, M, S, P)




# Plotten Sie zur Kontrolle wie in main2.py angegeben Histogramme der Verteilungen.

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
 
        
 
#######################################################################################################################
#######################################################################################################################

###   Thema: Gesetz der großen Zahlen (GdgZ)

def rel_variance_of_mean(rng, distr, no_samples, no_runs):
    
    #  X is a one-dimensional array with no_samples * no_runs elements, 
    # and the reshape method is used to transform it into a two-dimensional array with no_samples rows and no_runs columns.
    
    X = distr(rng, no_samples*no_runs).reshape(no_samples, no_runs)    
    
    # Hier Berechnunge der Mittelwerte über die Spalten von X: n array of length no_runs with the mean value for each sample
    
    Mittelwerte = np.mean(X, axis=0)
    
    # Berechnung der Varianzen: single value
    
    Varianzen = np.var(Mittelwerte, axis=0, ddof=1) 

    
    # Hier ist es zweckmäßig die Daten in einer Matrix anzuordnen
    # Varianz der Gesamtstichprobe --> 1 Dimension. Single value
    
    X_var = np.var(X.flatten())
    
      
    # Rückgabewert der Funktion ist der Quotient gebildet aus dieser 
    # Varianz und der Varianz der Gesamtstichprobe der Größe no_samples · no_runs. 
    
    return Varianzen / X_var




def Aufgabe_GdgZ():
    """Hier sollen die relativen Varianzen in Abhängigkeit von der Anzahl der 
    Samples (no_samples) geplottet werden."""
    no_runs = 100
    
    no_samples_max = 100 # Ich habe das eingefügt
    
    # Als Plotbefehle könnten Sie folgendes nutzen:
        
   
    
    for distr in distributions:
        
        rel_Varianzen= []
        
        for no_samples in range(1, no_samples_max + 1):
            
        
            rel_Varianzen.append(rel_variance_of_mean(rng, distr, no_samples, no_runs))
        
    
        plt.plot(range(1, no_samples_max + 1), rel_Varianzen, label=distr.__name__, alpha=0.5)
    
    plt.legend()
    plt.xlabel('Number of samples')
    plt.ylabel('Relative variance')
    plt.show()


#######################################################################################################################
#######################################################################################################################

###   Thema: Zentraler Grenzwertsatz (ZGWS)

def centralized_sample(rng, distr, no_samples, no_runs):
    
    # Matrix deren Spalten Stichproben der Größe no_samples
    X = distr(rng, no_samples*no_runs).reshape(no_samples, no_runs) # Im Skript: Sn = X1 + X2 + X3
    #print('X',X)
      
    
    # Zn = (Sn - n*µ) / (sqrt(n)*σ)
    

    
    # Sn := X1 + X2 + X3 +...+ Xn
    # Berechnung der zentralisierten Zufallsvariable aus den Spalten von X:  Ergebnis ist dann eine Stichprobe
    # der Größe no_runs von solchen Zufallsvariable
    Sn = np.sum(X, axis=0)   # Beispiel: array([2, 4, 6, ...no_runs])
    #print("Sn:", Sn)
    
    
    S = distr(rng,10**4)  # Samples
    
    # Mean value: Mittelwert µ: an array([5, 7, 2, ...no_runs])
    µ = np.mean(S, axis=0)
    #print("µ:", µ)
    
    
    # Standardabweichung
    σ = np.std(S, axis=0) # Here an array, with one element per column of X. One-dimensional array with no_runs elements
    #print("σ:", σ)
    
    
    # zentralisierten Zufallsvariable
    
    # he line np.tile(µ, (no_samples, 1)) creates an array with no_samples rows and 1 column, 
    # where each element of the column is equal to the mean value µ of the corresponding column of X.
    Zn = (Sn - no_samples*µ) / (np.sqrt(no_samples)*σ) # array
    
    #print("Zn:", Zn)
    
    
    return Zn    







def Aufgabe_ZGWS():
    no_runs = 10**6
    n = 3
    N = np.arange(1, n+1)
    bins = 100
    for distr in distributions:
        for n in N:
            X = centralized_sample(rng, distr, n, no_runs)
            plt.hist(X, bins=bins, density=True, histtype='stepfilled', alpha=0.6, label=distr.__name__ + f', n={n}');
        plt.legend()
        plt.show()
        

#######################################################################################################################
#######################################################################################################################

###   Ausführen der Bearbeitungen
        
Aufgabe_Bayes()
Aufgabe_Verteilungen()
Aufgabe_GdgZ()
Aufgabe_ZGWS()
