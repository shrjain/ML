# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:53:59 2013

@author: shrainik
"""
#!/usr/bin/python
from __future__ import division
from numpy import *
import matplotlib.pyplot as plt

add_ones = lambda a: hstack((ones((a.shape[0])).reshape(a.shape[0], 1),a))
path_prefix = 'clickprediction_data/'

def prob_Yequal1_givenWX(W,Xi):
    expnt = exp(Xi.dot(W))
    return expnt/(1+expnt)

def gradientDescent(X, Y, initW, eta, lmbd, num_of_iterations):
    newW = copy(initW)
    lenW = len(initW)
    N = X.shape[0]
    log_loss = []
    log_loss.append((lmbd/2.0)*(linalg.norm(newW[1:])**2) - sum(Y*(X.dot(newW)) - log(1+exp(X.dot(newW))))/N)
    
    for i in range(num_of_iterations):
        yj_minus_p = Y - prob_Yequal1_givenWX(newW, X)
        newW[0] = newW[0] + (eta/N)*sum(yj_minus_p)
        newW[1:] = (1 - eta*lmbd)*newW[1:] + eta*(array(yj_minus_p).dot(X[:,1:])/N)
        log_loss.append((lmbd/2.0)*(linalg.norm(newW[1:])**2) - sum(Y*(X.dot(newW)) - log(1+exp(X.dot(newW))))/N)
    
    return newW, log_loss

def gradientDescentStop(X, Y, initW, eta, lmbd, epsilon):
    newW = copy(initW)
    lenW = len(initW)
    N = X.shape[0]
    log_loss = []
    
    log_loss.append((lmbd/2.0)*(linalg.norm(newW[1:])**2) - sum(Y*(X.dot(newW)) - log(1+exp(X.dot(newW))))/N)
    num_of_iterations = 0
    convergence = False
    
    while convergence == False:
        yj_minus_p = Y - prob_Yequal1_givenWX(newW, X)
        newW[0] = newW[0] + (eta/N)*sum(yj_minus_p)
        newW[1:] = (1 - eta*lmbd)*newW[1:] + eta*(array(yj_minus_p).dot(X[:,1:])/N)
        
        log_loss.append((lmbd/2.0)*(linalg.norm(newW[1:])**2) - sum(Y*(X.dot(newW)) - log(1+exp(X.dot(newW))))/N)
        
        convergence = abs(log_loss[len(log_loss)-1] - log_loss[len(log_loss)-2]) < epsilon
        num_of_iterations += 1
        
    return newW, log_loss, num_of_iterations

if __name__ == '__main__':
    all_data = genfromtxt(path_prefix+'train.txt', delimiter=',')
    Y = all_data[:,0]
    X = add_ones(all_data[:, 1:])
    Xtest = add_ones(genfromtxt(path_prefix+'test.txt', delimiter=','))
    Ytest = genfromtxt(path_prefix+'test_label.txt')
    initW = zeros(X.shape[1])
    num_iteration = 1000
    
    newW, log_loss = gradientDescent(X,Y,initW,0.1, 0.3, num_iteration)
    print log_loss
    
    plt.plot(array(xrange(num_iteration)), log_loss[1:], 'g.--')
    plt.xlabel('iteration ->')
    plt.ylabel('log loss ->')
    plt.show()
    y_predicted = []
    for i in range(Xtest.shape[0]):
        y_predicted.append(1 if prob_Yequal1_givenWX(newW, Xtest[i,:]) > 0.5 else 0)
    
    SSE = sum((array(y_predicted)-Ytest)**2)
    print "SSE for batch gradient descent with 1000 iterations: ", SSE
    
    newW, log_loss, num_iteration = gradientDescentStop(X,Y,initW,0.1, 0.3, 0.0005)
    print "Number of iterations with the stopping criteria: ", num_iteration
    
    plt.plot(array(xrange(num_iteration)), log_loss[1:], 'g.--')
    plt.xlabel('iteration ->')
    plt.ylabel('log loss ->')
    plt.show()
    y_predicted = []
    for i in range(Xtest.shape[0]):
        y_predicted.append(1 if prob_Yequal1_givenWX(newW, Xtest[i,:]) > 0.5 else 0)
    
    SSE = sum((array(y_predicted)-Ytest)**2)
    print "SSE for batch gradient descent with a stop criteria: ", SSE
    