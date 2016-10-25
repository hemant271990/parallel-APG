import numpy as np
import theano
import theano.tensor as T
from theano import pp
from numpy import linalg as LA

d = 4
N = 20
eta = 0.05
gama_0 = 1.0
MAX_ITER = 20
A_in = np.random.normal(0, 1, (N,d))
b_in = np.random.normal(0, 0.01, (N,1))
#print [row[1] for row in s]
#y = np.random.choice([0,1], size=(N,1), p=[0.5, 0.5]) generate 0 or 1 with 0.5 prob each
W_0 = np.random.random((d, 1))
#print A_in
#print b_in
#print W_0

A = T.dmatrix("A")
b = T.dmatrix("b")
W = T.dmatrix('W')
Z = T.dmatrix("Z")
Aw = T.dot(A,W)

term1 = T.dot((T.transpose(Aw)), Aw)
term2 = 2*(T.dot(T.transpose(Aw), b))
term3 = T.dot(T.transpose(b), b)
l = ((1/2)*(term1 - term2 + term3)).sum()
grad_l = T.grad(l, W)

g = T.sum(abs(W))

grad_func = theano.function([A,b,W], grad_l, mode='DebugMode')
g_func = theano.function([W], g)

w_t = W_0
w_t_prev = w_t
w_t_next = w_t
u_t = W_0
u_t_prev = u_t
u_t_next = u_t
gama_t = gama_0
gama_t_prev = gama_t
gama_t_next = gama_t

for t in range (1, MAX_ITER):
	z_t = u_t - eta*grad_func(A_in, b_in, u_t)
	print [j for j in z_t]
	for i in range(0,d):
		w_t[i][0] = max((np.sign(z_t)[i] * ((np.absolute(z_t)) - np.full((d,1), eta))[i]), 0)
	
	gama_t_next = (1 + np.sqrt(1 + 4*gama_t ** 2))/2
	u_t_next = w_t + np.multiply((gama_t_prev/gama_t_next),  (w_t - w_t_prev))
	print [k for k in w_t]
	print "=================="

	w_t_prev = w_t
	w_t = w_t_next
	u_t_prev = u_t
	u_t = u_t_next
	gama_t_prev = gama_t
	gama_t = gama_t_next
