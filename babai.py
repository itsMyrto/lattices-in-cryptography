import numpy as np 
import math
from lll import LLL
from functions import gram_schmidt, inner_product

def babai(lattice, t):
    lll_reduced = LLL(lattice, delta=3/4)
    size = lll_reduced.shape[0]
    orthogonal_basis = gram_schmidt(lll_reduced)
    b = t
    for j in range(size-1, -1, -1):
        c = round(inner_product(b, orthogonal_basis[j]) / (inner_product(orthogonal_basis[j], orthogonal_basis[j])))
        b -= c*lll_reduced[j]
    return t - b 