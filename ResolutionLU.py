# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:11:01 2021

@author: Alexis
"""

import numpy as np
import decomposition_LU
import ResolutionSystTriInferieur
import ResolutionSystTriSup
import time
import matplotlib.pyplot as plt

def ResolutionLU(A,B):
    (L,U)=decomposition_LU.decomposition_LU(A)
    Aaug = np.concatenate((L,B), axis=1)
    n,m = np.shape(Aaug)
    Y = np.reshape(ResolutionSystTriInferieur.ResolutionSystTriInferieur(Aaug),(n,1))
    Aaug = np.concatenate((U,Y), axis=1)
    X = ResolutionSystTriSup.ResolutionSystTriSup(Aaug)
    return X

B = np.array([[7], [12] , [3]])
A = np.array([[2 , 5 , 6],[4 , 11 , 9],[-2 , -8 , 7]]) 


#print(ResolutionLU(A,B))
"""
temps = []
indices = []

for n in range(100,1000,50):
    
    A = np.random.rand(n,n)
    B = np.random.rand(n,1)
    
    temps_debut = time.time() 
    (L,U)=decomposition_LU.decomposition_LU(A)
    ResolutionLU(L,U,B)
    temps_fin = time.time() 
    temps_operation = temps_fin - temps_debut
    
    temps.append(temps_operation)
    indices.append(n)
    print("a")
    
abscisse =  indices
ordonnee =  temps
plt.plot(abscisse, ordonnee, label="LU")
plt.title("resolution LU")
plt.xlabel("taille matrice")
plt.ylabel("temps en seconde")

plt.legend()

plt.show()
"""