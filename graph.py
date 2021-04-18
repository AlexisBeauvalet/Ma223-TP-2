# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:02:45 2021

@author: Alexis
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import ResolCholesky
import np_linalg_cholesky
import ResolutionLU
import Gauss 

def erreurs(A,B,X):
   
    erreur = 0                
    n,m = A.shape
    matrice_erreur = (np.dot(A,X) - B)
    vect_erreur = np.ravel(matrice_erreur)
    for i in vect_erreur:
        erreur += abs(i)            
    erreur = erreur/n
    print("n:",erreur)
    return(erreur)


indices = []
temps = []
temps2 = []
temps3 = []
temps4 = []
temps5 = []

all_erreur_gauss = []
all_erreur_np_linalg_cholesky = []
all_erreur_Cholesky = []
all_erreur_ResolutionLU = []
all_erreur_linalg_solve = []

for n in range(100,800,50):
    
    B = np.random.rand(n,1)
    M = np.random.rand(n,n)
    A = np.dot(M,M.T)
    
    temps_debut = time.time() 
    X1=ResolCholesky.ResolCholesky(A,B)
    temps_fin = time.time() 
    erreur_Cholesky = erreurs(A,B,X1)
    temps_operation = temps_fin - temps_debut

    temps_debut2 = time.time() 
    X2=np_linalg_cholesky.np_linalg_cholesky(A,B)
    temps_fin2 = time.time() 
    erreur_np_linalg_cholesky = erreurs(A,B,X2)
    temps_operation2 = temps_fin2 - temps_debut2
    
    temps_debut3 = time.time() 
    X3 = ResolutionLU.ResolutionLU(A,B)
    temps_fin3 = time.time() 
    erreur_ResolutionLU = erreurs(A,B,X3)
    temps_operation3 = temps_fin3 - temps_debut3
    
    temps_debut4 = time.time() 
    X4 = np.linalg.solve(A,B)
    temps_fin4 = time.time() 
    erreur_linalg_solve = erreurs(A,B,X4)
    temps_operation4 = temps_fin4 - temps_debut4
    
    temps_debut5 = time.time() 
    X5 = Gauss.Gauss(A,B)
    temps_fin5 = time.time() 
    erreur_Gauss = erreurs(A,B,X5)
    temps_operation5 = temps_fin5 - temps_debut5

    temps.append(temps_operation)
    temps2.append(temps_operation2)
    temps3.append(temps_operation3)
    temps4.append(temps_operation4)
    temps5.append(temps_operation5)
    
    all_erreur_gauss.append(erreur_Gauss)
    all_erreur_np_linalg_cholesky.append(erreur_np_linalg_cholesky)
    all_erreur_Cholesky.append(erreur_Cholesky)
    all_erreur_ResolutionLU.append(erreur_ResolutionLU)
    all_erreur_linalg_solve.append(erreur_linalg_solve)
    indices.append(n)
    print("a")
    
abscisse =  indices
ordonnee =  temps
ordonnee2 =  temps2
ordonnee3 =  temps3
ordonnee4 =  temps4
ordonnee5 =  temps5

plt.plot(abscisse, ordonnee, label="resolution cholesky")
plt.plot(abscisse, ordonnee3, label="LU")
plt.plot(abscisse, ordonnee2, label="np.linalg.cholesky")
plt.plot(abscisse, ordonnee4, label="np.linalg.solve")
plt.plot(abscisse, ordonnee5, label="Gauss")

plt.title("comparaison du temps de calcul en fonction de la taille n")
plt.xlabel("taille matrice")
plt.ylabel("temps en seconde")

plt.legend()
plt.show()

abscisse = indices
graph_2_ordonnee = all_erreur_gauss
graph_2_ordonnee2 = all_erreur_Cholesky
graph_2_ordonnee3 = all_erreur_ResolutionLU
graph_2_ordonnee4 = all_erreur_np_linalg_cholesky
graph_2_ordonnee5 = all_erreur_linalg_solve

plt.plot(abscisse,graph_2_ordonnee,label="Gauss")
plt.plot(abscisse,graph_2_ordonnee2,label = "Cholesky")
plt.plot(abscisse,graph_2_ordonnee3,label = "LU")
plt.plot(abscisse,graph_2_ordonnee4,label = "np.linalg.cholesky")
plt.plot(abscisse,graph_2_ordonnee5,label = "np.linalg.solve")

plt.title("Erreur de calcul en fonction de la taille n")
plt.xlabel("taille matrice")
plt.ylabel("nombre d'erreurs")


plt.legend()
plt.show()



