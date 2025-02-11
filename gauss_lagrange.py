import math
import random
import numpy as np
from functions import inner_product

def norm(u):
    return math.sqrt(inner_product(u, u))

def compute_angle(u, v):
    normu = norm(u)
    normv = norm(v)
    v_inner_u = inner_product(v, u)
    cos_theta = v_inner_u / (normu * normv)
    angle = math.acos(cos_theta)
    return math.degrees(angle)

def generate_random_vector():
    while True:
        cord1 = random.randint(-100000, 100000)
        cord2 = random.randint(-100000, 100000)
        if cord1 == 0 and cord2 == 0:
            continue
        else:
            break
    return np.array([cord1, cord2])

def gauss_lagrange_basis_reduction(b_0, b_1):
    while True:
        if norm(b_0) > norm(b_1):
            temp = b_0.copy()
            b_0 = b_1.copy()
            b_1 = temp.copy()  
        
        numerator = inner_product(b_0, b_1)
        denominator = inner_product(b_0, b_0)
        m = round(numerator / denominator)
        if m == 0:
            break
        b_1 = b_1 - m * b_0

    return b_0, b_1

def experiment():
    sum_angle = 0
    ITERATIONS = 50
    for i in range(0, ITERATIONS):
        u = generate_random_vector()
        v = generate_random_vector()
        u, v = gauss_lagrange_basis_reduction(u, v)
        angle = compute_angle(u, v)
        sum_angle += angle

    average_angle_degrees = sum_angle / ITERATIONS
    print("Average angle is ", average_angle_degrees)