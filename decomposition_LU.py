# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:54:27 2021

@author: Alexis
"""

import numpy as np

def decomposition_LU(A):
    n,m = np.shape(A)
    L = np.eye(n)
    U = A
    
    for i in range(0,n-1):
        
        for j in range(i+1,n):
            pivot= U[j,i]/U[i,i]
            L[j,i]= pivot
            
            for k in range(i,n):
                U[j,k]= U[j,k] - pivot * U[i,k]
    
    return L,U

A = np.array([[2,-2,4],[1,-3,1],[3,7,5]])

(L,U)=decomposition_LU(A)
