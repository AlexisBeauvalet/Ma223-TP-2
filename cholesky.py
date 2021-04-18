# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:48:42 2021

@author: Alexis
"""
import numpy as np


def cholesky(A):
    n , m = np.shape(A)
    L= np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,i+1):
            somme1 = 0 
            somme2 = 0 
            for k in range(0,i):
                somme1 = somme1 + (L[i,k])**2 
                somme2 = somme2 +  L[i,k]*L[j,k]
            L[i,i] = np.sqrt(A[i,i]-somme1)
            L[i,j] = (A[i,j] - somme2) / L[j,j]        
    return L
    


A = np.array([[4 , -2 , -4],[-2 , 10 , 5],[-4 , 5 , 6]]) 
B = np.array([[6], [-9] , [-7]])

#print("\n",cholesky(A))



#np.linalg.solve(A,B)
