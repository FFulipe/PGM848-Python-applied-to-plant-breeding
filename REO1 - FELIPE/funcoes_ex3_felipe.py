# EXERCÍCIO 03:
# Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral um vetor qualquer,
# baseada em um loop (for).

import numpy as np

def media (vetor):
    soma = 0
    it = 0
    for i in vetor:

        soma += i # somador = somador + i
        it+=1 # it = it+1
    mean = soma/it
    return mean

def var_amostral (vetor):
    soma = 0
    it = 0
    sdq = 0
    for i in vetor:

        soma += i  # soma = soma + i
        it += 1  # it = it+1
        sdq += i**2 #sdq = sdq + i**2

    var = (sdq - ((soma**2/it)))/(it-1)
    return var
