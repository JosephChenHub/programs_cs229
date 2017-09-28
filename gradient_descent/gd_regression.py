#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

#batch gradient descent and Stochastic gradient descent implementation for Linear Regression
#h(x) = W'*x
#J(W) = 0.5\sum_{i=1}^{m} (h - y)^2,here m is the number of samples.

#---global parameters ---
lr = 0.0001  #learning rate
D = 2 	   #number of variables,plus 1(W0)
#W = np.zeros(D)	  #parameters
W = np.random.normal(0,0.001,D)  
M = 250 	  #number of samples
iterations = 100  #steps
N = 3 	   #degree of the polynomial function
batch_size = 10

mu = 0
sigma = 1.5
x1 = np.linspace(0,100,M) 
y = 3.5*x1 + 2.7 + np.random.normal(mu,sigma,M)
X = np.array([np.ones(M),x1])

total_loss = []	
print 'initial W:',W
#---define the hypothesis ---
def h(x,W):
	return W.dot(x) #here x=(x1,x2,...) is a point
#---cost function ---
def J(x,y,W):
	out = 0
	for i in range(0,M): #for all instances
		temp = (h(x[:,i],W) - y[i])*(h(x[:,i],W)-y[i])
		out = out + temp
	return np.log(out)

#---batch gradient descent ---
def BGD(x,y,W):
	del total_loss[:]
	for it in range(iterations): #steps
		for j in range(D):   
			delta = 0
			for i in range(M):
				delta += (y[i] - h(x[:,i],W))*x[j,i]
			W[j] += lr*delta*1.0/M		
		loss = J(x,y,W)
		total_loss.append(loss)	
		print 'iter:'+str(it)+' loss:'+ str(loss)+' w:' + str(W)
		print '\r\n'

#---stochastic gradient descent ---
def SGD(x,y,W):
	del total_loss[:]
	for it in range(iterations): #steps
		for j in range(D):   
			delta = 0
			for i in range(M):
				delta = (y[i] - h(x[:,i],W))*x[j,i]
				W[j] += lr*delta		

		loss = J(x,y,W)	
		total_loss.append(loss)
		print 'iter:'+str(it)+' loss:'+ str(loss)+' w:' + str(W)
		print '\r\n'
#---mini-batch gradient descent ---
def mini_BGD(x,y,W):	
	del total_loss[:]
	if batch_size >= M:
		BGD(x,y,W)
	else:
		for it in range(iterations): #iter
			for j in range(D):   #each variable
				t = M/batch_size
				remain = M - batch_size*t
				for k in range(t):
					delta = 0
					for i in range(batch_size): #batch
						delta += (y[k*batch_size+i] - h(x[:,k*batch_size+i],W))*x[j,k*batch_size+i]
					W[j] += lr*delta*1.0/batch_size
					
				delta = 0
				for i in range(remain):
					delta += (y[t*batch_size+i] - h(x[:,t*batch_size+i],W))*x[j,t*batch_size+i]

				W[j] += lr*delta*1.0/batch_size
			loss = J(x,y,W)
			total_loss.append(loss)	
			print 'iter:'+str(it)+' loss:'+ str(loss)+' w:' + str(W)
			print '\r\n'

#using BGD
BGD(X,y,W)

#W.reshape(D,1)
hat = W[0]*X[0,:].reshape(1,M) + W[1]*X[1,:].reshape(1,M)
plt.subplot(321)
plt.title('batch gradient descent')
plt.grid()
plt.plot(x1,y,'g*')
plt.plot(x1,hat.reshape(M,1),'r')
plt.subplot(322)
plt.title('training loss')
plt.grid()
plt.plot(total_loss)
#SGD
del total_loss[:]
W = np.random.normal(0,0.001,D)  
SGD(X,y,W)

hat = W[0]*X[0,:].reshape(1,M) + W[1]*X[1,:].reshape(1,M)
plt.subplot(323)
plt.title('stochastic gradient descent')
plt.grid()
plt.plot(x1,y,'g*')
plt.plot(x1,hat.reshape(M,1),'r')
plt.subplot(324)
plt.title('training loss')
plt.grid()
plt.plot(total_loss)

#mini-BGD
del total_loss[:]
W = np.random.normal(0,0.001,D)  
mini_BGD(X,y,W)

hat = W[0]*X[0,:].reshape(1,M) + W[1]*X[1,:].reshape(1,M)
plt.subplot(325)
plt.title('mini-batch gradient descent')
plt.grid()
plt.plot(x1,y,'g*')
plt.plot(x1,hat.reshape(M,1),'r')
plt.subplot(326)
plt.title('training loss')
plt.grid()
plt.plot(total_loss)

plt.show()

'''
z = np.matrix(y -  hat.reshape(1,M)) 
mean = np.mean(z)
cov = np.cov(z)
print 'mean:',mean
print 'cov:',cov
'''

