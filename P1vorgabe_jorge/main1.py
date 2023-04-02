# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math

# mittel(x): Berechnung und Rueckgabe des arithmetischen Mittels der Zahlen in x. Dabei soll x ein iterable,
# also etwa eine Liste oder ein numpy-Vector con Zahalen sein

def mittel(x):
    
    x = np.array(x)
    n = len(x)

    x_mittel = sum(x)/n
    
    return x_mittel



# Berechnung des Quartils gemaes Vorlesung. Dabei ist x gegeben wie oben
# und p eine Zahl zwischen 0 und 1

def quantil(x,p):
    
    x = np.array(x)
    
    # Funktion sortiert von klein nach groß
    x.sort()
    
    n = len(x)
    
    if n == 1 and  0<= p <= 1:
        
        x_p = x[0]

    elif n != 1 and p == 1:

        x_p = x[-1]        
        
    else:
    
        # naechstkleinere Ganzzahl
        # Im Python Indixierung faengt bei null
        o = int(p*(n - 1))
    
        x_p = (1 - p)*x[o] + p*x[o + 1]
    
    
    return x_p



# Berechnung des Medians

def median(x):
    
    x = np.array(x)
    
    # Funktion sortier von klein nach groß
    x.sort()
    
    n = len(x)
    
    # n gerade    
    if (n % 2) == 0:
        
        # Im Python Indixierung faengt bei null
        # // --> integer
        x_median = (x[n//2 - 1] + x[n//2 + 1 - 1])/2
        
    # n ungerade
    else:
        
        # Im Python Indixierung faengt bei null
        x_median = x[(n + 1)//2 - 1]
    
    
    return x_median



# Berechnung der unkorrigierten Stichprobevarianz

def var(x):
    
    x = np.array(x)
    
    n = len(x)
    
    x_mittel = mittel(x)
    
        
    var = sum((x - x_mittel)**2)/n
        
    
    return var
    

    
# Rueckgabewert soll ein Dreituppel sein, bestehend aus Steigung
# und Achsenabschnitt der Regressionsgeraden sowie dem quadratischen Fehler

def regress(x, y):
    
    x = np.array(x)
    y = np.array(y)

    
    n = len(x)
    
    # Empirische Kovarianz
    
    s_xy = sum(((x - mittel(x))*(y - mittel(y))))/n
    
       
    # Steigung
    
    B = s_xy/var(x)


    # Achsenabschnitt
    
    alfa = mittel(y) - B*mittel(x)
    
    
    # Quadratische Fehler
        
    Q = sum((y - (alfa + B*x))**2)

    
    return (B, alfa, Q)



# Rueckgabewert soll ein Dreitupel sein, bestehend aus der Transformationsmatrix
# Q, dem Vektor der Eigenwerte (in fallender Reihenfolge), sowie
# dem transformierten Datensatz in Matrixform

def pca(X):
    
    X = np.array(X)
    
    n = len(X)
    
    # X transponiert
    X_T = np.transpose(X)
    
    mittelt_X_T = []
    
    for X_T_i in X_T:
        
        mittelt_X_T.append(mittel(X_T_i))
    
    
    mittelt_X = np.array([mittelt_X_T for e in range(len(X))]) # Selbe Dimension wie B

        
    
    B = X - mittelt_X
    
    B_T = np.transpose(B)
    
    # Kovarianzmatrix
    C = B_T.dot(B)/(n - 1)
    
    
    # Berechnung der Eigenwerte und Eigenvektoren
    eig_vals, eig_vecs = np.linalg.eig(C)
    
    # Eigenwert-Eigenvektor-Paarung
    # Gesucht sind die Spalten vom Eingenvektor
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    
    # Sortieren der Eigenwerte und -vektoren absteigend
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Konstruktion der Transformationsmatrix Q aus den sortierten Eigenvektoren
    Q = np.column_stack([e[1] for e in eig_pairs])
    
    # Transformieren des zentrierten Datensatzes mit der Transformationsmatrix Q
    X_transformed = B.dot(np.transpose(Q))
    
    return Q, eig_vals, X_transformed




if __name__ == '__main__':
    pass
    
    
    
    