# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np


# mittel(x): Berechnung und Rueckgabe des arithmetischen Mittels der Zahlen in x. Dabei soll x ein iterable,
# also etwa eine Liste oder ein numpy-Vector con Zahalen sein

def mittel(x):
    
    x = np.array(x)
    n = len(x)

    x_mittel = np.sum(x)/n
    
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
    
    
    return np.quantile(x,p)



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
    
        
    var = np.sum((x - x_mittel)**2)/n
        
    
    return var
    

    
# Rueckgabewert soll ein Dreituppel sein, bestehend aus Steigung
# und Achsenabschnitt der Regressionsgeraden sowie dem quadratischen Fehler

def regress(x, y):
    
    x = np.array(x)
    y = np.array(y)

    
    n = len(x)
    
    # Empirische Kovarianz
    
    s_xy = np.sum(((x - mittel(x))*(y - mittel(y))))/n
    
       
    # Steigung
    
    B = s_xy/var(x)


    # Achsenabschnitt
    
    alfa = mittel(y) - B*mittel(x)
    
    
    # Quadratische Fehler
        
    Q = np.sum((y - (alfa + B*x))**2)

    
    return (B, alfa, Q)



# Rueckgabewert soll ein Dreitupel sein, bestehend aus der Transformationsmatrix
# Q, dem Vektor der Eigenwerte (in fallender Reihenfolge), sowie
# dem transformierten Datensatz in Matrixform

def pca(X):
    
    X = np.array(X)
    
    #n = len(X)
    
    # X transponiert
    #X_T = np.transpose(X)
    
    #mittelt_X_T = []
    
    #for X_T_i in X_T:
        
        #mittelt_X_T.append(mittel(X_T_i))
    
    
    #mittelt_X = np.array([mittelt_X_T for e in range(len(X))]) # Selbe Dimension wie B

        
    # Zentrieren der Daten
    #B = X - mittelt_X
    B = X - X.mean(axis=0)
    
    B_T = B.T
    
    # Kovarianzmatrix
    C = B_T@B/(len(B) - 1)
    
    # QR-Zerlegung
    # Q, R = np.linalg.qr(C)
    # D ist die Diagonalmatrix der Eigenwerte
    # D = np.diag(np.diag(np.dot(R, Q)))

    # Eigenwertzerlegung
    # D ist die Diagonalmatrix der Eigenwerte
    # Q die Matrix der Eigenvektoren
    D, Q = np.linalg.eig(C)
    
    
    # Suche nach den richtigen Index
    idx = D.argsort()[::-1]
    
    # Eigenwert-Eigenvektor-Paarung
    # Gesucht sind die Spalten vom Eingenvektor
    D, Q = D[idx], Q[:,idx]
    
    # Transformieren des zentrierten Datensatzes
    X_transformed = B@Q
    
    return Q, D, X_transformed




if __name__ == '__main__':
    
    #eig_pairs = [(np.abs(D[i]), Q[:,i]) for i in range(len(D))]
    
    # Sortieren der Eigenwerte und -vektoren absteigend
    #eig_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Konstruktion der Transformationsmatrix Q aus den sortierten Eigenvektoren
    #Q = np.column_stack([e[1] for e in eig_pairs])
    
    pass
    
    
    
    