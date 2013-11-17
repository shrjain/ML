# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:39:20 2013

@author: shrainik
"""
import numpy as np
import matplotlib.pyplot as plt

def kernelizedPerceptron(X, y, kernel):
    mistakes = []
    predictions = np.array(np.ones(len(y)))
    for i in range(len(y)):
        prediction_i = 0
        for mistake in mistakes:
            prediction_i += y[mistake]*kernel(X[mistake], X[i])
        
        if prediction_i < 0:
            predictions[i] = -1
        
        if predictions[i] != y[i]:
            mistakes.append(i)
            
    return predictions, mistakes
        
if __name__ == '__main__':
    path_prefix = '/Users/shrainik/Dropbox/ML/ML/HW3/'
    data = np.genfromtxt(path_prefix + 'validation.csv', delimiter = ',', skip_header = 1)
    y_train = data[:,0]
    X_train = data[:,1:]
    
    kernel_k1_p = lambda u,v: np.dot(u,v) + 1
    
    predictions,mistakes = kernelizedPerceptron(X_train, y_train, kernel_k1_p)
    
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
    
    
    d = np.array([1,3,5,7,10,15,20])
    loss_function = np.array(d, dtype = np.float) #just creating an array of length same as d, irrespective of values, will set values later
    count = 0
    for degree in d:
        kernel_kd_p = lambda u,v: ((np.dot(u,v) + 1)**degree)
        
        predictions, mistakes = kernelizedPerceptron(X_train, y_train, kernel_kd_p)
        loss_function[count] = (1.0/1000)*len(mistakes)
        count +=1
    
    plt.plot(d, loss_function)
    plt.xlabel('d')
    plt.ylabel('Average Loss after 1000 iterations')
    plt.show()
    
    
    dataTest = np.genfromtxt(path_prefix + 'test.csv', delimiter = ',', skip_header = 1)
    y_test = dataTest[:,0]
    X_test = dataTest[:,1:]    
    
    list_loss =  list(loss_function)
    best_d = d[list_loss.index(min(list_loss))]
    kernel_gauss_1000 = lambda u,v: np.exp((-1.0)*((np.linalg.norm(u-v)**2)/(2.0*1000000)))
    
    
    kernel_bestd = lambda u,v: ((np.dot(u,v) + 1)**best_d)
    
    predictions1,mistakes1 = kernelizedPerceptron(X_test, y_test, kernel_bestd)
    predictions2,mistakes2 = kernelizedPerceptron(X_test, y_test, kernel_gauss_1000)
    
    loss_function1 = np.zeros(len(y_train))
    mistCount = 0
    for i in range(len(y_test)):
        if i in mistakes1:
            mistCount+=1
        loss_function1[i] = (1.0/(i+1))*mistCount
    
    loss_function2 = np.zeros(len(y_train))
    mistCount = 0
    for i in range(len(y_test)):
        if i in mistakes2:
            mistCount+=1
        loss_function2[i] = (1.0/(i+1))*mistCount

    anchorPoints = np.array(range(1,11))*100
    plt.plot(anchorPoints, loss_function1[anchorPoints - 1])    
    plt.plot(anchorPoints, loss_function2[anchorPoints - 1])
    plt.legend(('Polynomial, d=3','Gaussian, sigma = 1000'), loc='best')
    plt.xlabel('Iteration Count')
    plt.ylabel('Average Loss')
    plt.show()
    