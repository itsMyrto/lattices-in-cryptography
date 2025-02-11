import numpy as np 
import math
from functions import gram_schmidt, inner_product

def kfp_enumeration(lattice, R):
    S = []
    n = lattice.shape[0]
    m = lattice.shape[1]
    l = np.zeros(n)
    x = np.zeros(n)
    c = np.zeros(n)
    B_ = gram_schmidt(lattice)
    B = np.zeros((n))
    mij = np.zeros((n, m))

    for i in range(0, n):
        B[i] = inner_product(B_[i], B_[i])
    
    for i in range(0, n):
        for j in range(0, n):
            mij[j][i] = inner_product(lattice[j], B_[i]) / inner_product(B_[i], B_[i]) 

    i = 0 # The indexing is from 0 to n-1, not 1 to n like in the pseudocode
    while i < n:
        # Compute ci: - Sum from j=i+1 to n: xj*μji
        c[i] = 0
        for j in range(i+1, n):
            c[i] += x[j] * mij[j][i]
        c[i] = -c[i]

        # Compute li: ||bi*||^2 * (xi-ci)^2 
        l[i] = B[i] * (x[i] - c[i]) ** 2

        # Compute sumli: l[i] + l[i+1] + ... + l[n]
        sumli = 0
        for j in range(i, n):
            sumli += l[j]
                    
        if sumli <= R**2:
            if i == 0: # Not i==1 due to different indexing
                # Sum from j=0 to n xj*bj
                vec = x[0] * lattice[0]
                for j in range(1, n):
                    vec += x[j] * lattice[j]
                # Append the vector in the S set
                S.append(vec)
                x[0] = x[0] + 1
            else:
                i = i - 1
                # Computing Sum from j=i+1 to n lj
                sum_l = 0
                for j in range(i+1, n):
                    sum_l += l[j]

                # Computing Sum from j=i+1 to n μji * xj 
                sum_mx = 0
                for j in range(i+1, n):
                    sum_mx += x[j] * mij[j][i]

                # The left part of the interval
                x[i] = math.ceil(-sum_mx - math.sqrt((R**2 - sum_l) / B[i]))
        else:
            i = i + 1
            if i == n:
                break
            x[i] = x[i] + 1

    return S 


def schnorr_euchner(lattice, R, a):
    n = lattice.shape[0]
    m = lattice.shape[1]
    B_ = gram_schmidt(lattice)
    B = np.zeros((n))
    mij = np.zeros((n, m))

    for i in range(0, n):
        B[i] = inner_product(B_[i], B_[i])
    
    for i in range(0, n):
        for j in range(0, n):
            mij[j][i] = inner_product(lattice[j], B_[i]) / inner_product(B_[i], B_[i]) 

    x = np.zeros(n)
    c = np.zeros(n)
    l = np.zeros(n)
    delta_x = np.zeros(n)
    delta_square_x = np.full(n, -1)
    
    S = []
    i = 0

    delta_x[0] = 1
    delta_square_x[0] = 1

    while i < n:
        c[i] = 0
        for j in range(i+1, n):
            c[i] += x[j] * mij[j][i]
        c[i] = -c[i]
    
        l[i] = B[i] * (x[i] - c[i]) ** 2

        sumli = 0
        for j in range(i, n):
            sumli += l[j]
        
        if sumli <= (a[n-1] * R)**2 and i == 0:
            vec = x[0] * lattice[0]

            for j in range(1, n):
                vec += x[j] * lattice[j]
            
            S.append(vec)
            
        if sumli <= (a[n-i-1] * R)**2 and i > 0:
            i = i - 1
            
            c[i] = 0
            for j in range(i+1, n):
                c[i] += x[j] * mij[j][i]
            c[i] = -c[i]

            x[i] = round(c[i])
            delta_x[i] = 0

            if c[i] < x[i]:
                delta_square_x[i] = 1
            else:
                delta_square_x[i] = -1
        else:
            i += 1
            if i == n:
                break
            delta_square_x[i] = -delta_square_x[i]
            delta_x[i] = -delta_x[i] + delta_square_x[i]
            x[i] = x[i] + delta_x[i]

    return S


