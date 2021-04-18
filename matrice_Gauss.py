import numpy as np


def ReductionGauss(Aaug):
    nbLigne = Aaug.shape[0]
    nbColone = Aaug.shape[1]
    
    if(nbColone != nbLigne + 1):
        return "matrice n'est pas augmentée."

    for i in range (nbColone - 1):
        if(Aaug[i, i] != 0):
            for j in range(i + 1, nbLigne):
                Aaug[j] = Aaug[j] - (Aaug[j, i] / Aaug[i, i]) * Aaug[i]        
        else:
            return(" pivot nul")

    return(Aaug)

matrice = np.array([[2 , 5 , 6],[4 , 11 , 9],[-2 , -8 , 7]]) 
result = np.array([[7], [12] , [3]])
matriceAug = np.append(matrice, result, axis = 1)

#print(ReductionGauss(matriceAug))

