# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:03:37 2024

@author: Victoria Johnson
         v.johnson@sheffield.ac.uk
         Automatic Control and Systems Engineering 
         The University of Sheffield
"""

import numpy as np
from scipy import optimize
import time 

def sub1(x):
    x1 = x[2]
    z = x[0:2]
    y2_hat = x[4]
    y1 = z[0]**2 + z[1] + x1 - 0.2*y2_hat
    return y1
    
def sub2(x):
    z = x[0:2]
    y1_hat = x[3]
    if y1_hat <= 0:
        y1_hat = 1e-8
    y2 = y1_hat**0.5 + z[0] + z[1]
    return y2

def objective(x):
    z = x[0:2]
    x1 = x[2]
    f = x1**2 + z[1] + sub1(x) + np.exp(-sub2(x))
    return f

def g1(x):
    return -(3.16 - sub1(x))

def g2(x):
    return -(sub2(x) - 24)

def h1(x):
    return (sub1(x) - x[3]) - 1e-6

def h2(x):
    return (sub2(x) - x[4]) - 1e-6

bnds = ((-10, 10), (0, 10), (0, 10), (3.16, None), (None, 24))    # Bounds
x0 = [5, 2, 1, 1.0, 1.0]    # Initial conditions

# Constraint tuple...
constraints = ({'type': 'ineq', 'fun': g1},
               {'type': 'ineq', 'fun': g2},
               {'type': 'eq', 'fun': h1},
               {'type': 'eq', 'fun': h2})


t0 = time.time()

# Perform optimisation
res = optimize.minimize(objective, 
                  x0, 
                  bounds = bnds, 
                  constraints=constraints,
                  method='SLSQP')
tf = time.time()
x = res.x

print(res.message)
print('Optimisation took: ')
print('\t'+ str(tf - t0) + ' seconds')
print('\t'+ str(res.nit) + ' iterations')
print('\t'+ str(res.nfev) + ' function evaluations')

print('F = ' + str(objective(x)))
print('')
print('z = ' + str(x[0:2]))
print('x = ' + str(x[2]))
print('y1_hat = ' + str(x[3]))
print('y2_hat = ' + str(x[4]))
print('y1 = ' + str(sub1(x)))
print('y2 = ' + str(sub2(x)))

print('')
print('g1(x) = ' + str(g1(x)))
print('g2(x) = ' + str(g1(x)))
print('h1(x) = ' + str(h1(x)))
print('h2(x) = ' + str(h2(x)) + '\n')

print('Error = ' + str(abs(res.fun - 3.18339)/3.18339))







