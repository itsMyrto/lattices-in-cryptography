import numpy as np 
import math

def inner_product(v, u):
    return np.inner(v, u)

def projection_operator(v, u):
    vu = inner_product(v, u)
    uu = inner_product(u, u)
    return (vu / uu) * u

def gram_schmidt(lattice):
    size = lattice.shape[0]
    new_lattice = [None] * size
    new_lattice[0] = lattice[0].copy()

    for i in range(1, size):
        new_lattice[i] = lattice[i].copy()
        for j in range(0, i):
            new_lattice[i] -= projection_operator(lattice[i], new_lattice[j])
    
    return np.array(new_lattice)


