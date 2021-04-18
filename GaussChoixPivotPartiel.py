# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:16:56 2021

@author: Alexis
"""

import numpy as np
import ResolutionSystTriSup
import time
import matplotlib.pyplot as plt

def GaussChoixPivotPartiel(A,B):
    Aaug = np.concatenate((A,B), axis=1)
    n,m = np.shape(Aaug)
    
    for i in range(n):
        for j in range(0,n-i):
            if abs(Aaug[j,j]) < abs(Aaug[i+j,j]) :
                i_max = Aaug[i,:].copy()
                Aaug[i,:] = Aaug[i+j,:]
                Aaug[i+j,:] = i_max
                #print(Aaug)
            for k in range(i+1,n):
                g = Aaug[k,i]/Aaug[i,i]
                Aaug[k,:] = Aaug[k,:] - g * Aaug[i,:]
                #print(Aaug)
    #print(Aaug)          
    solution= ResolutionSystTriSup.ResolutionSystTriSup(Aaug)
    #print(solution)
    return solution 


A = np.array([[2 , 2 , 1],[10 , 0 , -2],[-2 , 7 , 0]], float)
B = np.array([[0], [4] , [-9]], float )

GaussChoixPivotPartiel(A,B)


temps = []
indices = []

for n in range(100,1000,50):
    
    A = np.random.rand(n,n)
    B = np.random.rand(n,1)
    
    temps_debut = time.time() 
    GaussChoixPivotPartiel(A,B)
    temps_fin = time.time() 
    temps_operation = temps_fin - temps_debut
    
    temps.append(temps_operation)
    indices.append(n)
    print("a")
    
abscisse =  indices
ordonnee =  temps
plt.plot(abscisse, ordonnee, label="pivot partielle")
plt.title("pivot partielle")
plt.xlabel("taille matrice")
plt.ylabel("temps en seconde")

plt.legend()

plt.show()
            