# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:10:48 2024

@author: Victoria Johnson
         v.johnson@sheffield.ac.uk
         Automatic Control and Systems Engineering 
         The University of Sheffield
"""

import numpy as np
from scipy import optimize
import time 

global fevals
fevals = 0
def sub1(z, x1, y):
    return z[0]**2 + z[1] + x1 - 0.2*y[1] - y[0]

def sub2(z, x1, y):
    if y[0] <= 0:
        y[0] = 1e-8
    return y[0]**0.5 + z[0] + z[1] - y[1]


def f(z, x1, y):
    F = np.array([sub1(z, x1, y),
                  sub2(z, x1, y)])
    return F
    
def jacobian(y):
    if y[0] == 0:
        y[0] = 1e-8
    
    J = np.array([[-1, -0.2],[(1/(1/(2*y[0]**0.5))).item(), -1]])
    return J
    
def MDA(x):
    global fevals
    fevals = fevals + 1
    nIts = 20
    z = x[0:2]
    x1 = x[2]
    x0 = np.array([[1.0], [1.0]])
    archive = np.zeros((nIts, 2))
    for m in range(0, nIts):
        J_eval = jacobian(x0)
        inv_J = np.linalg.inv(J_eval)
        f_eval = f(z, x1, x0)
        x0 = x0 - inv_J@f_eval
        archive[m, :] = x0.reshape((1, 2))
        if m > 1:
            if archive[m, 0] - archive[m - 1, 0] < 1e-8:
                break
    return x0

def objective(x):
    z = x[0:2]
    x1 = x[2]
    y = MDA(x)
    f = x1**2 + z[1] + y[0] + np.exp(-y[1])
    return f

def g1(x):
    y = MDA(x)
    return -(3.16 - y[0])

def g2(x):
    y = MDA(x)
    return -(y[1] - 24)

x = np.array([1.9776, 0.0, 0.0])

bnds = ((-10, 10), (0, 10), (0, 10))    # Bounds
x0 = [5, 2, 1]    # Initial conditions
# Constraint tuple...
constraints = ({'type': 'ineq', 'fun': g1},
               {'type': 'ineq', 'fun': g2})

t0 = time.time()
res = optimize.minimize(objective, 
                  x0, 
                  bounds = bnds, 
                  constraints=constraints,
                  method='SLSQP')
tf = time.time()
x = res.x
y = MDA(x)
print(res.message)
print('Optimisation took: ')
print('\t'+ str(tf - t0) + ' seconds')
print('\t'+ str(res.nit) + ' iterations')
print('\t'+ str(res.nfev) + ' function evaluations')

print('F = ' + str(objective(x)))
print('')
print('z = ' + str(x[0:2]))
print('x = ' + str(x[2]))
print('y1 = ' + str(y[0]))
print('y2 = ' + str(y[1]))

print('')
print('g1(x) = ' + str(g1(x)))
print('g2(x) = ' + str(g1(x)) + '\n')

print('Error = ' + str(abs(res.fun - 3.18339)/3.18339))

print('MDA called ' + str(fevals) + ' times')
















