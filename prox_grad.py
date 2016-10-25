import numpy as np
import theano
import theano.tensor as T
from theano import pp
from numpy import linalg as LA

d = 4
N = 20
eta = 0.05
MAX_ITER = 20
A_in = np.random.normal(0, 1, (N,d))
b_in = np.random.normal(0, 0.01, (N,1))
#print [row[1] for row in s]
#y = np.random.choice([0,1], size=(N,1), p=[0.5, 0.5]) generate 0 or 1 with 0.5 prob each
W_0 = np.random.random((d, 1))
#print A_in
#print b_in
#print W_0
#A_in = np.matrix([[1.0, 2.0], [3.0, 4.0]])
#b_in = np.matrix([[2.0], [2.0]])
#W_0 = np.matrix([[2.0], [2.0]])
#print A_in
#print b_in
#print W_0

A = T.dmatrix("A")
b = T.dmatrix("b")
W = T.dmatrix('W')
Z = T.dmatrix("Z")
Aw = T.dot(A,W)
#print s

term1 = T.dot((T.transpose(Aw)), Aw)
term2 = 2*(T.dot(T.transpose(Aw), b))
term3 = T.dot(T.transpose(b), b)
l = ((1/2)*(term1 - term2 + term3)).sum()
grad_l = T.grad(l, W)

g = T.sum(abs(W))

grad_func = theano.function([A,b,W], grad_l, mode='DebugMode')
g_func = theano.function([W], g)
#print grad_func(A_in, b_in, W_0)

w_t = W_0
for t in range (0, MAX_ITER):
	z_t = w_t - eta*grad_func(A_in, b_in, w_t)
	print [j for j in z_t]
	tmp = np.zeros(d)
	for i in range(0,d):
		tmp[i] = np.sign(z_t)[i] * ((np.absolute(z_t)) - np.full((d,1), eta))[i]
	w_t = [[max(0, i)] for i in tmp]
	print [k for k in w_t]
	print "=================="
