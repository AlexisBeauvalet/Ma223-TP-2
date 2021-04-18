# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:06:18 2021

@author: Alexis
"""


import numpy as np
import ResolutionSystTriSup

def ReductionGaussChoixPivotTotal(A):
    Aaug = np.concatenate((A,B), axis=1)
    n,m = np.shape(Aaug)
    
    for i in range(n):
        for j in range(0,n-i):
            if abs(Aaug[j,j]) < abs(Aaug[i+j,j]) :
                i_max = Aaug[i,:].copy()
                Aaug[i,:] = Aaug[i+j,:]
                Aaug[i+j,:] = i_max
            if abs(Aaug[j,j]) < abs(Aaug[j,i+j]) :
                J_max = Aaug[:,i].copy()
                Aaug[:,i] = Aaug[:,i+j]
                Aaug[:,i+j] = J_max
            for k in range(i+1,n):
                g = Aaug[k,i]/Aaug[i,i]
                Aaug[k,:] = Aaug[k,:] - g * Aaug[i,:]
    return Aaug


def GaussChoixPivotTotal(A,B):
    Aaug = np.concatenate((A,B), axis=1)
    
    Taug = ReductionGaussChoixPivotTotal(Aaug)
    solution= ResolutionSystTriSup.ResolutionSystTriSup(Taug)
    
    return solution

A = np.array([[2 , 5 , 0],[4 , 11 , 9],[-2 , -8 , 7]])
B = np.array([[7], [12] , [3]])

print(GaussChoixPivotTotal(A,B))

