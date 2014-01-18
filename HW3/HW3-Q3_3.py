# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:37:19 2013

@author: shrainik
"""
import numpy as np
import matplotlib.pyplot as plt

def svmStochastic (X, Y, eta, C):
    W = np.zeros(X.shape[1])
    mistakes  = []
    W0 = 0
    for i in range(len(Y)):
        error = 1 - Y[i]*(np.dot(W,X[i])+W0)
        if error > 0 :
            mistakes.append(i)
        W = W + eta*(C*(1 if error > 0 else 0)*np.dot(Y[i],X[i]) - 2*W)
        W0 = W0 + eta*C*Y[i]        
        
    return W0,W, mistakes

if __name__ == '__main__':
    path_prefix = '/Users/shrainik/Dropbox/ML/ML/HW3/'
    data = np.genfromtxt(path_prefix + 'validation.csv', delimiter = ',', skip_header = 1)    
    y_train = data[:,0]
    X_train = data[:,1:]
    data = np.genfromtxt(path_prefix + 'test.csv', delimiter = ',', skip_header = 1)
    y_test = data[:,0]
    X_test = data[:,1:]
    
    W0,W,mistakes = svmStochastic (X_test, y_test, 0.00001, 1)
    
    loss_function = np.zeros(len(y_train))
    mistCount = 0
    for i in range(len(y_train)):
        if i in mistakes:
            mistCount+=1
        loss_function[i] = (1.0/(i+1))*mistCount
    anchorPoints = np.array(range(1,11))*100
    plt.plot(anchorPoints, loss_function[anchorPoints - 1])
    plt.xlabel('Iteration Count')
    plt.ylabel('Average Loss')
    plt.show()