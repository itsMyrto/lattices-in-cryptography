import numpy as np 
import math
from lll import LLL
from functions import inner_product

def orthogonal_lattice(vec):
    n = 1
    d = len(vec)
    B = vec.copy()

    prod_norm = round(math.sqrt(inner_product(vec, vec)))

    c = math.ceil(2**((d-1)/2 + (d-n)*(d-n-1)/4) * prod_norm)

    I = np.eye(B.shape[0])
    c_B = c * B
    B_ = np.block([[c_B], [I]])

    L = LLL(B_.T, 0.99)

    # We take only the last d entries of each vector and we keep only the first d-n (4-1 = 3) vectors 
    basis_orthogonal = np.array([L[i][-d:] for i in range(d-n)])
    print("Orthogonal Basis:\n", basis_orthogonal)

    print("--------------------------------Testing------------------------------------")
    print("We multiply the vector with every basis vector. We want all entries to be 0")
    print("---------------------------------------------------------------------------")
    for v in basis_orthogonal:
        print(inner_product(v, vec))