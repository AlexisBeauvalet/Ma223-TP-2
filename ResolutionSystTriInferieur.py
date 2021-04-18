# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:21:14 2021

@author: Alexis
"""

import numpy as np

def ResolutionSystTriInferieur(Taug):
    
    n , m = np.shape(Taug)
    
    Y= np.zeros((n), dtype=int)
    for i in range (0,n):
        somme=0
        for j in range(0,n):
            somme= somme + Y[j]*Taug[i,j]
        Y[i]=(Taug[i,n]-somme)/Taug[i,i]
    return(Y)