from gauss_lagrange import gauss_lagrange_basis_reduction, experiment, compute_angle
from lll import LLL, lll_experiment, lllvsgauss
from babai import babai
from orthogonal import orthogonal_lattice
from enumeration import kfp_enumeration, schnorr_euchner
import numpy as np
import math
from knapsack import build_lattice_basis, solve_knapsack
from functions import inner_product

np.set_printoptions(
    formatter={'int': '{:2d}'.format}
)


# # # # # # # # # # # # # # # # # # # # # # GAUSS LAGRANGE # # # # # # # # # # # # # # # # # # # # # #
# ----------------------------------------------------------------------------------------------------
# To run the Gauss Lagrange reduction method with custom parameters use this line of code:
# Some examples, including the ones in the project's pdf / Uncomment to run:
# 
# print(gauss_lagrange_basis_reduction(np.array([1, 1]), np.array([3, 4])))
# print(gauss_lagrange_basis_reduction(np.array([-1, 1]), np.array([300, 400])))
# print(gauss_lagrange_basis_reduction(np.array([66586820, 65354729]), np.array([6513996, 6393464])))
# print(gauss_lagrange_basis_reduction(np.array([0, 1020301]), np.array([1, 8239876])))
# print(gauss_lagrange_basis_reduction(np.array([6, 5]), np.array([-6, 6])))
# -----------------------------------------------------------------------------------------------------
# To run the experiment with the average angle between the vectors after reduction uncomment this:
#
# experiment()
# -----------------------------------------------------------------------------------------------------
# To Compute the angle between two vectors uncomment this:
#
# print(compute_angle(np.array([-1,0]),np.array([0,1])))
# print(compute_angle(np.array([-1,1]),np.array([350,350])))
# -----------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # GAUSS LAGRANGE # # # # # # # # # # # # # # # # # # # # # #



# # # # # # # # # # # # # # # # # # # # # # # # # LLL # # # # # # # # # # # # # # # # # # # # # # # # #
# ----------------------------------------------------------------------------------------------------
# Some examples on how to run the LLL algorithm:
#
# b0 = np.array([-1, -1, -2, -4, -1])
# b1 = np.array([-2, -2, 4, -4, 2])
# b2 = np.array([4, 2, -1, 1, 4])
# b3 = np.array([1, -1, 1, -1, -1])
# b4 = np.array([-1, 1, -1, 1, -30])
# basis_matrix = np.array([b0, b1, b2, b3, b4], float)
# lll_reduced = LLL(basis_matrix, 3/4)
# print(lll_reduced)
#
# b0 = np.array([42, -36,   -5])
# b1 = np.array([ -13,  9,  30])
# basis_matrix = np.array([b0, b1], float)
# lll_reduced = LLL(basis_matrix, 3/4)
# print(lll_reduced)

# b0 = np.array([-6, 5, -2, -6])
# b1 = np.array([-9, -1, 8, 0])
# b2 = np.array([-10, -3, -9, -8])
# b3 = np.array([-1, -3, 3, -9])
# basis_matrix = np.array([b0, b1, b2, b3], float)
# lll_reduced = LLL(basis_matrix, 0.99)
# print(lll_reduced)
# -----------------------------------------------------------------------------------------------------
# To run the experiment and check if the LLL produces the correct λ1:
#
# lll_experiment()
#
# -----------------------------------------------------------------------------------------------------
# LLL vs Gauss Lagrange in 2D bases
#
# for i in range(0, 10):
#     lllvsgauss()
# -----------------------------------------------------------------------------------------------------
# # # # # # # # # # # # # # # # # # # # # # # # # LLL # # # # # # # # # # # # # # # # # # # # # # # # #



# # # # # # # # # # # # # # # # # # # # # # # # # BABAI # # # # # # # # # # # # # # # # # # # # # # # # 
# Examples on how to run the babai algorithm: 
# 
# b0 = np.array([137, 312])
# b1 = np.array([215, -187])
# basis_matrix = np.array([b0, b1], float)
# print(babai(basis_matrix, [53172, 81743]))

# b0 = np.array([1, 2, 3])
# b1 = np.array([3, 0, -3])
# b2 = np.array([3, -7, 3])
# basis_matrix = np.array([b0, b1, b2], float)
# print(babai(basis_matrix, [10, 6, 5]))

# b0 = np.array([7, 0, 1])
# b1 = np.array([1, 17, 1])
# b2 = np.array([-3, 0, 10])
# basis_matrix = np.array([b0, b1, b2], float)
# print(babai(basis_matrix, [100, 205, 305]))
#
# # # # # # # # # # # # # # # # # # # # # # # # # BABAI # # # # # # # # # # # # # # # # # # # # # # # # 



# # # # # # # # # # # # # # # # # # # # # ORTHOGONAL LATTICE # # # # # # # # # # # # # # # # # # # # # #     
# To find a lattice basis perpendicular to a vector run this:
#
# orthogonal_lattice(np.array([125, -75, 45, -27]))
#
# # # # # # # # # # # # # # # # # # # # # ORTHOGONAL LATTICE # # # # # # # # # # # # # # # # # # # # # #     


# # # # # # # # # # # # # # # # # # # # # # KFP ENUMERATION # # # # # # # # # # # # # # # # # # # # # # #     
# To run the KFP enumeration:
#
# b0 = np.array([-1, -1, -2, -4, -1])
# b1 = np.array([-2, -2, 4, -4, 2])
# b2 = np.array([4, 2, -1, 1, 4])
# b3 = np.array([1, -1, 1, -1, -1])
# b4 = np.array([-1, 1, -1, 1, -30])
# basis_matrix = np.array([b0, b1, b2, b3, b4], float)
# print(kfp_enumeration(basis_matrix, math.sqrt(20)))

# b0 = np.array([3, 6, 13])
# b1 = np.array([11, 3, 15])
# b2 = np.array([12, 12, 0])
# basis_matrix = np.array([b0, b1, b2], float)
# print(kfp_enumeration(basis_matrix, math.sqrt(214)))
#
# # # # # # # # # # # # # # # # # # # # # # KFP ENUMERATION # # # # # # # # # # # # # # # # # # # # # # #     



# # # # # # # # # # # # # # # # # # # # # # SCHNORR EUCHNER # # # # # # # # # # # # # # # # # # # # # # #     
# To run the Schnorr-Euchner enumeration:
#
# b0 = np.array([-1, -1, -2, -4, -1])
# b1 = np.array([-2, -2, 4, -4, 2])
# b2 = np.array([4, 2, -1, 1, 4])
# b3 = np.array([1, -1, 1, -1, -1])
# b4 = np.array([-1, 1, -1, 1, -30])
# basis_matrix = np.array([b0, b1, b2, b3, b4], float)
# n = basis_matrix.shape[0]
# a = np.zeros(n)
# for i in range(0, n):
#     a[i] = min(1, math.sqrt(((i+1) * 1.05 / n)))
# print(schnorr_euchner(basis_matrix, math.sqrt(40), a))
#
# # # # # # # # # # # # # # # # # # # # # # SCHNORR EUCHNER # # # # # # # # # # # # # # # # # # # # # # #     


# # # # # # # # # # # # # # # # # # # # SOLVING KNAPSACK PROBLEMS # # # # # # # # # # # # # # # # # # # #     
#
# B, pos, sol = solve_knapsack()
# if pos != None:        
#     b = B[pos]
#     x = []
#     n = len(b) - 3
#     for i in range(n):
#         x.append(int(abs(b[i]-b[n+1])/2))
    
#     print(f"b is {b} and found at position {pos} of the lattice")
#     print(f"x solution is {x}")
#     print(f"checking if solution is correct: s=x*a={int(inner_product(x, np.array(sol[0])))} and the real s is {sol[1]}")
# else:
#     print("Failed to find x")
#
# # # # # # # # # # # # # # # # # # # # SOLVING KNAPSACK PROBLEMS # # # # # # # # # # # # # # # # # # # #     
