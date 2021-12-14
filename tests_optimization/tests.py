# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:25:22 2021

@author: sega01
"""
import pickle
from cvxopt import solvers, matrix, spdiag, log, exp, div
import numpy as np
import matplotlib.pyplot as plt
solvers.options['show_progress'] = True

n=2
e=np.abs(np.matrix([0.5,0.2,-0.4,0.8,0.6,0.1,-0.3,0.7]))
phi=np.matrix([[0.8,2],[0.1,1.2],[0.3,0.9],[0.6,3],[0.3,1.6],[0.4,2.1],[0.1,0.7],[0.6,2]])

# https://cvxopt.org/userguide/coneprog.html#quadratic-programming

# Problem data.
n = phi.shape[1]
k =phi.shape[0]
z_l=n+k
sigma=1
alpha=0.01
beta=0.00001

# the base option ... but -t<phi*x<t (where t is forced to be bigger than the error) allows for x to be 0
def option_A():
    # q must allow that only the t elements are represented in the minimization (so far) ... P can be used later for regularization
    p_com=np.zeros([z_l,z_l])
    p_com[:n,:n]=np.identity(n)*alpha
    p_com[n:,n:]=np.identity(k)*beta
    P = matrix(p_com)
    q_com=np.zeros([z_l,1])
    q_com[n:,:]=1
    q = matrix(q_com)
    # G will contain three parts: 1st for the error bounds and then the two limits for the absolute value conversion through t
    g_com=np.zeros([3*k,z_l])
    g_com[:k,n:]=-sigma*np.identity(k)
    g_com[k:2*k,:n]=phi
    g_com[k:2*k,n:]=-np.identity(k)
    g_com[2*k:3*k,:n]=-phi
    g_com[2*k:3*k,n:]=-np.identity(k)
    G = matrix(g_com)
    # h is just the error and a lot of 0s
    h_com=np.zeros([3*k,1])
    h_com[:k,:]=-e.T
    h=matrix(h_com)
    # no linear equalities so far ...
    A = matrix(0.0, (3*k,z_l))
    b = matrix(0.0, (3*k,1))
    # from numpy.linalg import matrix_rank
    # matrix_rank([P,A, G])
    res = solvers.qp(P, q, G, h) #, A, b
    print(res['x'])
    z=np.array(list(res['x']))
    x=z[:n]
    t=z[n:]
    
    # see weights
    sq_w=0.5*np.matmul(np.matmul(z.T,p_com),z)
    pr_w=np.matmul(q_com.T,z)[0]
    t_w=sq_w+pr_w
    print('Minimization results --> Total: {} | Square weights: {} | Projection width: {}'.format(t_w,sq_w,pr_w))
    print('The value of Phi*X is: {}'.format(np.matmul(phi,x)))
    print('The value of T is: {}'.format(t))
    return res

# instead of imposing the limit on t, we pose it on phi*x>|e| --> forces x to project always on possitive values 
def option_B():
    # q must allow that only the t elements are represented in the minimization (so far) ... P can be used later for regularization
    p_com=np.zeros([z_l,z_l])
    p_com[:n,:n]=np.identity(n)*alpha
    p_com[n:,n:]=np.identity(k)*beta
    P = matrix(p_com)
    q_com=np.zeros([z_l,1])
    q_com[n:,:]=1
    q = matrix(q_com)
    # G will contain three parts: 1st for the error bounds and then the two limits for the absolute value conversion through t
    g_com=np.zeros([3*k,z_l])
    g_com[:k,:n]=-sigma*phi
    g_com[k:2*k,:n]=phi
    g_com[k:2*k,n:]=-np.identity(k)
    g_com[2*k:3*k,:n]=-phi
    g_com[2*k:3*k,n:]=-np.identity(k)
    G = matrix(g_com)
    # h is just the error and a lot of 0s
    h_com=np.zeros([3*k,1])
    h_com[:k,:]=-e.T
    h=matrix(h_com)
    # no linear equalities so far ...
    A = matrix(0.0, (3*k,z_l))
    b = matrix(0.0, (3*k,1))
    # from numpy.linalg import matrix_rank
    # matrix_rank([P,A, G])
    res = solvers.qp(P, q, G, h) #, A, b
    print(res['x'])
    z=np.array(list(res['x']))
    x=z[:n]
    t=z[n:]
    
    # see weights
    sq_w=0.5*np.matmul(np.matmul(z.T,p_com),z)
    pr_w=np.matmul(q_com.T,z)[0]
    t_w=sq_w+pr_w
    print('Minimization results --> Total: {} | Square weights: {} | Projection width: {}'.format(t_w,sq_w,pr_w))
    print('The value of Phi*X is: {}'.format(np.matmul(phi,x)))
    print('The value of T is: {}'.format(t))
    return res

res=option_B()
z=np.array(list(res['x']))
x=z[:n]
t=z[n:]
fig = plt.figure(figsize=(15.0, 10.0))
xs=np.linspace(0,k-1,k)
plt.plot(xs, np.ravel(np.matmul(phi,x)).flatten(), "o", markersize=4, color='blue', alpha=0.5)
plt.plot(xs, np.ravel(np.matmul(phi,x)).flatten(), "-", linewidth=2, color='silver', alpha=0.6,label='error estimation')
plt.plot(xs, np.ravel(e), "o", markersize=4, color='red', alpha=0.5)
plt.plot(xs, np.ravel(e), "-", linewidth=2, color='gold', alpha=0.6,label='real error')
plt.legend()