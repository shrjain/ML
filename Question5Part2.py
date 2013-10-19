# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:41:55 2013

@author: shrainik
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.io as io


def generateData(N,d,k,sigma):
	X = np.random.randn(N,d)
	w_act = np.matrix(np.zeros(d)).T
	for i in range(k):
		wi = 10 if np.random.randint(1,3) == 1 else -10
		w_act[i] = wi
		
	epsilon = sigma * np.random.randn(N,1)
	y = np.matrix(X)*w_act + epsilon
	return X,y,w_act

def computePrecisionRecall(k,d, W_HAT):
	preci = np.zeros(len(W_HAT))
	recall = np.zeros(len(W_HAT))
	for i in range(len(W_HAT)):
		correctZero = 0
		tZero = 0
		for j in range(d):
			if  W_HAT[i][0,j] != 0:
				if	j < k:
					correctZero +=1
				tZero +=1
		if tZero == 0:
			preci[i] = 1
		else:
			preci[i] = correctZero*1.0/tZero
		recall[i] = correctZero*1.0/k
	return preci, recall

def lasso_solver(X,wi,w0i,y,lmbd):
	convergence = False
	delta = 0.2
	currentIteration = 0
	maxIterations = 3000
	n = len(X.T[0])
	d = len(wi)
	w = wi
	w_old = wi
	w0 = w0i
	w0_old = w0i
	while convergence == False and currentIteration < maxIterations:
		currentIteration += 1
		w_old = w
		w0_old = w0
		for j in range(d):
			w_dum = w
			w_dum[j] = 0
			aj = 2*((X.T[j]**2).sum())
			jth_col = np.reshape(X[:,j],[n,1])
			cj = np.dot(jth_col.T,y - w0 - np.dot(X,w_dum))
			if	cj < lmbd * -1:
				wj_hat = (cj+lmbd)/aj
			elif cj > lmbd:
				wj_hat = (cj-lmbd)/aj
			else:
				wj_hat = 0
			w[j] = wj_hat
		w0 = (1.0/len(y))*(sum(y*1.0-(np.matrix(X)*np.matrix(w)).getA()))
		convergence = True
		for i in range(d):
			if np.absolute(w[i] - w_old[i])/w_old[i] > delta:
				convergence = False 
				break
		if convergence == True:
			if np.absolute(w0_old - w0)/w0_old > delta:
				convergence = False
	return w0,w

def syntheticDataPart1(N=50,d=75,k=5,sigma=1,factor=1.15):
	X,y,w_act = generateData(N,d,k,sigma)
	y_bar = np.mean(y)
	w_int = np.matrix(np.zeros(d)).T
	lambda_max = 2* max(np.absolute(np.matrix(X).T*(y-y_bar)))
	
	w_hat1 = lasso_solver(X,w_int,0,y,lambda_max)
	lambdaStore = np.array(lambda_max)
	lamda = lambda_max/factor
	W_HAT = w_hat1[1].T
	while lamda > 1:
		lambdaStore = np.hstack((lambdaStore,lamda))
		w_hat1 = lasso_solver(X,w_hat1[1],w_hat1[0],y,lamda)
		W_HAT = np.vstack((W_HAT, w_hat1[1].T))
		lamda = lamda/factor
	
	precision,recall = computePrecisionRecall(k,d,W_HAT)
	print precision, recall
	
	print lambdaStore.getA()
	plt.plot(lambdaStore.getA()[0], precision, 'r.--')
	plt.plot(lambdaStore.getA()[0], recall, 'yo--')
	plt.xlabel('Lambda')
	plt.legend(('Precision', 'Recall'),loc='best')
	plt.show()
	return W_HAT

def syntheticDataPart2(lambdasArray, N=50,d=75,k=5,sigma=10):
	X,y,w_act = generateData(N,d,k,sigma)
	w_int = np.matrix(np.zeros(d)).T

	w_hat1 = lasso_solver(X,w_int,0,y,lambdasArray[0])
	W_HAT = w_hat1[1].T
	
	for iteration in range(1,len(lambdasArray)):
		w_hat1 = lasso_solver(X,w_int,0,y,lambdasArray[iteration])
		W_HAT = np.vstack((W_HAT, w_hat1[1].T))

	return W_HAT

if __name__ == "__main__":
	N=50
	d=75
	k=5
	sigma=1
#Part1 for Ques 5.2: 
	syntheticDataPart1(N,d,k,sigma,1.10)
#Par1Ends

#Part2 for Ques 5.2: 
	lambdaStoreBasedOnPreviousSolution = np.array([500,480,460,440,420,400,380,360,340,320])   
	for i in range(10):
		W_HatCollectionForLambdas = syntheticDataPart2(lambdaStoreBasedOnPreviousSolution, N,d,k,10)
		precision,recall = computePrecisionRecall(k,d,W_HatCollectionForLambdas)
		print precision, recall
		plt.plot(lambdaStoreBasedOnPreviousSolution, precision, 'r')
		plt.plot(lambdaStoreBasedOnPreviousSolution, recall, 'g')
	plt.xlabel('Lambda')
	plt.ylabel('Red lines are the precisions green lines are recall')
	plt.show()
#Part2Ends

#Part3 for 5.2    
	syntheticDataPart1(50,75,5,1,1.15)
	syntheticDataPart1(50,150,5,1,1.15)
	syntheticDataPart1(50,1000,5,1,1.15)
	syntheticDataPart1(100,75,5,1,1.15)
	syntheticDataPart1(100,150,5,1,1.15)
	syntheticDataPart1(100,1000,5,1,1.15)
##end Part3
