# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:59:41 2021

@author: Alexis
"""
import matrice_Gauss
import ResolutionSystTriSup
import numpy as np

def Gauss(A,B):
    Aaug= np.c_[A, B]
    Taug= matrice_Gauss.ReductionGauss(Aaug)
    #print(Taug , "\n")
    solution= ResolutionSystTriSup.ResolutionSystTriSup(Taug)
    #print(solution)
    return solution

matrice = np.array([[4 , -2 , -4],[-2 , 10 , 5],[-4 , 5 , 6]]) 
result = np.array([[6], [-9] , [-7]])

#print(Gauss(matrice,result))



