import numpy as np 
import math
import itertools
from functions import gram_schmidt, inner_product
from gauss_lagrange import gauss_lagrange_basis_reduction
from enumeration import kfp_enumeration

def LLL(lattice, delta):
    orthogonal_basis = gram_schmidt(lattice)
    size = lattice.shape[0]
    k = 1
    while k < size :
        for j in range(k-1, -1, -1):
            lattice[k] = lattice[k] - round(inner_product(orthogonal_basis[j], lattice[k]) / inner_product(orthogonal_basis[j], orthogonal_basis[j])) * lattice[j]
            orthogonal_basis = gram_schmidt(lattice)
        if inner_product(orthogonal_basis[k], orthogonal_basis[k]) >=  (delta - (inner_product(orthogonal_basis[k-1], lattice[k]) ** 2) / ((inner_product(orthogonal_basis[k-1], orthogonal_basis[k-1])) ** 2)) * inner_product(orthogonal_basis[k-1], orthogonal_basis[k-1]):
            k = k + 1
        else:
            temp = lattice[k].copy()
            lattice[k] = lattice[k-1].copy()
            lattice[k-1] = temp.copy()
            orthogonal_basis = gram_schmidt(lattice)
            k = max(k-1, 1)
    return lattice

def generate_full_rank_matrix(rows, cols):
    matrix = np.random.randint(-50, 50, size=(rows, cols)).astype(float)
    while np.linalg.matrix_rank(matrix) < rows:
        matrix = np.random.randn(rows, cols)
    return matrix

def lll_experiment(dim=4):
    ITERATIONS = 50
    count = 0
    for i in range(0, ITERATIONS):
        random_lattice = generate_full_rank_matrix(dim, dim)
        print("Original Basis\n", random_lattice)
        lll_reduced = LLL(random_lattice, 0.99)
        print("LLL reduced basis\n", lll_reduced)
        smallest_vector = lll_reduced[0]
        print("Smallest Vector by LLL: ", smallest_vector)
        norm_sv = math.sqrt(inner_product(smallest_vector, smallest_vector))
        print("Norm of the vector: ", norm_sv)

        S = kfp_enumeration(random_lattice, math.ceil(norm_sv))
        print(S)
        for vec in S:
            vec_norm = math.sqrt(inner_product(vec, vec))
            if vec_norm < norm_sv and vec_norm != 0:
                print("The smallest vector is ", vec)
                count += 1

    print(f"LLL failed {count} times out of {ITERATIONS} iterations to find the shortest vector")

def lllvsgauss():
    B = generate_full_rank_matrix(2, 2)
    B_ = B.copy()
    print("Original Basis:\n", B_)
    lll_reduced = LLL(B, 0.99)
    gauss_reduced = gauss_lagrange_basis_reduction(B_[0], B_[1])
    print("LLL reduced\n", lll_reduced, "\nGauss reduced\n", gauss_reduced)
    return


