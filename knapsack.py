import numpy as np
import random, operator
import math
from lll import LLL
from functions import inner_product

def dot_product(a,b):
    return sum(map( operator.mul, a, b))
    
def rand_bin_array(K, N):
    arr = np.array([0] * K + [1] * (N-K))
    np.random.shuffle(arr)
    return  arr

def find(n,d,hamming):
    if n%2==1:
        return "enter an even integer"
    a =  [random.randint(1,math.floor(2**((2-d)*n))) for _ in range(n)]         
    density=float(len(a)/math.log(max(a),2));
    solution = rand_bin_array(n-hamming,n)
    a0 = dot_product(solution,a);      
    aold=a;
    a0old=a0;
    return a,a0,density,sum(solution),len(solution),solution

def build_lattice_basis(n, d, hamming):
    N = random.randint(int(math.sqrt(n)) + 1, 10000)
    sol = find(n, d, hamming)
    B = 2 * np.eye(n)
    new_col = N*np.array(sol[0])
    B = np.column_stack((B, new_col))
    B = np.column_stack((B, np.zeros(n)))
    B = np.column_stack((B, N*np.ones(n)))
    last_row = np.ones(n+3)
    last_row[n] = N*sol[1]
    last_row[n+2] = hamming*N
    B = np.vstack((B, last_row))
    return B, sol

def verify_b(b, n):
    found_solution = True

    for j in range(n):
        if abs(b[j]) == 1:
            continue
        else:
            found_solution = False
            break

    if found_solution:
        if b[n] == b[n+2] == 0 and abs(b[n+1]) == 1:
            pass
        else:
            found_solution = False

    return found_solution

def solve_knapsack():
    n = 30 # Lattice dimensions
    d = 0.99 # Density
    hamming = 15 # Hamming distance
    B, sol = build_lattice_basis(n, d, hamming) # Building the matrix
    pos = None
    print("The original lattice for the knapsack is:\n", B.astype(int))

    for i in range(0, 15):
        solution_found = False
        permuted_indices = np.random.permutation(B.shape[0]) # Choosing a random permutation
        B = B[permuted_indices] # Applying the permutation to the rows
        B = LLL(B, 0.99) # Reduce with LLL

        for j in range(B.shape[0]):
            solution_found = verify_b(B[j], n)
            if solution_found:
                pos = j
                break
        
        if solution_found:
            break

        flag = False
        print("Rescaling...")
        while not flag:
            vector_norms = np.linalg.norm(B, axis=1)
            sorted_indices = np.argsort(vector_norms)
            B = B[sorted_indices] 

            for j in range(B.shape[0]-1):
                
                b_j_norm = math.sqrt(inner_product(B[j], B[j])) 

                for k in range(j-1):
                    new_vec1 = B[j] - B[k] 
                    new_vec2 = B[j] + B[k]

                    new_vec1_norm = math.sqrt(inner_product(new_vec1, new_vec1))
                    new_vec2_norm = math.sqrt(inner_product(new_vec2, new_vec2))
                    
                    if math.sqrt(inner_product(new_vec1, new_vec1)) < b_j_norm:
                        B[j] = new_vec1
                        flag = True
                    elif new_vec2_norm < b_j_norm:
                        B[j] = new_vec2
                        flag = True

            if flag:
                flag = False
            else:
                flag = True

    
    return B, pos, sol
            
                



