# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:36:58 2021

@author: Alexis
"""
import numpy as np
import ResolutionSystTriSup
import ResolutionSystTriInferieur
import cholesky 

def ResolCholesky(A,B):
    n , m = np.shape(A)
    n2, m2 = np.shape(B)
    L = cholesky.cholesky(A)
    L2 = L.T
    Laug = np.concatenate((L,B),axis = 1)
    #print(Laug)
    Y = np.reshape(ResolutionSystTriInferieur.ResolutionSystTriInferieur(Laug),(n2,1))
    #print(Y)
    T = np.concatenate((L2,Y),axis = 1)
    #print(T)
    X = ResolutionSystTriSup.ResolutionSystTriSup(T)
    return(X)
    
#A = np.array([[4 , -2 , -4],[-2 , 10 , 5],[-4 , 5 , 6]]) 
B = np.array([[6], [-9] , [-7]])

M = np.random.rand(3,3)
A = np.dot(M,M.T) 


#print("\n",ResolCholesky(A,B))