# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:31:19 2021

@author: Alexis
"""

import numpy as np

def ResolutionSystTriSup(Taug):
    
    n , m = np.shape(Taug)
    
    X= np.zeros(n)
    for i in range (n-1,-1,-1):
        somme=0
        for j in range(i,n):
            somme= somme + X[j]*Taug[i,j]
        X[i]=(Taug[i,n]-somme)/Taug[i,i]
    return(X)
            
    
Taug=np.array([[2,5,6,7],[0,1,-3,-2],[0,0,4,4]])

ResolutionSystTriSup(Taug)