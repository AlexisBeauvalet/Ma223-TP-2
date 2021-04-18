# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:49:05 2021

@author: Alexis
"""

import numpy as np
import Gauss
import time
import matplotlib.pyplot as plt


temps = []
indices = []

for n in range(100,1000,50):
    
    A = np.random.rand(n,n)
    B = np.random.rand(n,1)
    
    temps_debut = time.time() 
    Gauss.Gauss(A,B)
    temps_fin = time.time() 
    temps_operation = temps_fin - temps_debut
    
    
    temps.append(temps_operation)
    indices.append(n)
    
abscisse =  indices
ordonnee =  temps
plt.semilogy(abscisse, ordonnee, label="gauss")
plt.title("log y algorithme de Gauss")
plt.xlabel("taille matrice")
plt.ylabel("temps en seconde")

plt.legend()

plt.show()
