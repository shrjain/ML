# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:53:59 2013

@author: shrainik
"""
#!/usr/bin/python
from __future__ import division
from numpy import *
import time as time
import matplotlib.pyplot as plt

add_ones = lambda a: hstack((ones((a.shape[0])).reshape(a.shape[0], 1),a))
path_prefix = '/Users/shrainik/Downloads/hw2-data 2/clickprediction_data/'

def prob_Yequal1_givenWX(W,Xi):
    expnt = exp(Xi.dot(W))
    return expnt / (1 + expnt)

def stochasticDescent(X, Y, initW, eta, lmbd, number_iterations_around_data):
    newW = copy(initW)
    lenW = len(initW)
    N = X.shape[0]
    log_loss = []
    log_loss.append((lmbd / 2.0) * (linalg.norm(newW[1:]) ** 2) - sum(Y * (X.dot(newW)) - log(1 + exp(X.dot(newW)))) / N)
    for k in range(number_iterations_around_data):
        for i in range(N):
            yi_minus_p = Y[i] - (1-1/(1+exp(X[i,:].dot(newW))))
            newW[0] = newW[0] + (eta) * yi_minus_p
            newW[1:] = (1 - eta * lmbd) * newW[1:] + eta * (yi_minus_p * (X[i,1:]))
            log_loss.append((lmbd / 2.0) * (linalg.norm(newW[1:]) ** 2) - sum(Y * (X.dot(newW)) - log(1 + exp(X.dot(newW)))) / N)
    return newW, log_loss

def gradientDescent(X, Y, initW, eta, lmbd, num_of_iterations):
    newW = copy(initW)
    lenW = len(initW)
    N = X.shape[0]
    log_loss = []
    log_loss.append((lmbd / 2.0) * (linalg.norm(newW[1:]) ** 2) - sum(Y * (X.dot(newW)) - log(1 + exp(X.dot(newW)))) / N)
    
    for i in range(num_of_iterations):
        yj_minus_p = Y - prob_Yequal1_givenWX(newW, X)
        newW[0] = newW[0] + (eta / N) * sum(yj_minus_p)
        newW[1:] = (1 - eta * lmbd) * newW[1:] + eta * (array(yj_minus_p).dot(X[:,1:]) / N)
        log_loss.append((lmbd / 2.0) * (linalg.norm(newW[1:]) ** 2) - sum(Y * (X.dot(newW)) - log(1 + exp(X.dot(newW)))) / N)
    
    return newW, log_loss

def gradientDescentStop(X, Y, initW, eta, lmbd, epsilon):
    newW = copy(initW)
    lenW = len(initW)
    N = X.shape[0]
    log_loss = []
    
    log_loss.append((lmbd / 2.0) * (linalg.norm(newW[1:]) ** 2) - sum(Y * (X.dot(newW)) - log(1 + exp(X.dot(newW)))) / N)
    num_of_iterations = 0
    convergence = False
    
    while convergence == False:
        yj_minus_p = Y - prob_Yequal1_givenWX(newW, X)
        newW[0] = newW[0] + (eta / N) * sum(yj_minus_p)
        newW[1:] = (1 - eta * lmbd) * newW[1:] + eta * (array(yj_minus_p).dot(X[:,1:]) / N)
        
        log_loss.append((lmbd / 2.0) * (linalg.norm(newW[1:]) ** 2) - sum(Y * (X.dot(newW)) - log(1 + exp(X.dot(newW)))) / N)
        
        convergence = abs(log_loss[len(log_loss) - 1] - log_loss[len(log_loss) - 2]) < epsilon
        num_of_iterations += 1
        
    return newW, log_loss, num_of_iterations

def printPrecisionRecall(Ytest, y_predicted):
    A_for_class1 = len(nonzero(Ytest)[0])
    A_for_class0 = len(Ytest) - A_for_class1

    B_for_class1 = len(nonzero(array(y_predicted))[0])
    B_for_class0 = len(y_predicted) - B_for_class1

    A_intersection_B_Class1 = len(intersect1d(nonzero(Ytest)[0], nonzero(array(y_predicted))[0]))
    A_intersection_B_Class0 = len(intersect1d(where(Ytest == 0)[0], where(array(y_predicted) == 0)[0]))

    if B_for_class0 == 0:
        precision_0 = 0
    else:
        precision_0 = A_intersection_B_Class0/B_for_class0
    if B_for_class1 == 0:
        precision_1 = 0
    else:
        precision_1 = A_intersection_B_Class1/B_for_class1
    
    recall_0 = A_intersection_B_Class0/A_for_class0
    recall_1 = A_intersection_B_Class1/A_for_class1

    print "Precision and Recall for class 0:", precision_0, recall_0
    print "Precision and Recall for class 1:", precision_1, recall_1

if __name__ == '__main__':
    all_data = genfromtxt(path_prefix + 'train.txt', delimiter=',')
    Y = all_data[:,0]
    X = add_ones(all_data[:, 1:])
    Xtest = add_ones(genfromtxt(path_prefix + 'test.txt', delimiter=','))
    Ytest = genfromtxt(path_prefix + 'test_label.txt')
    initW = zeros(X.shape[1])

    newW_stochastic, log_loss_stochastic = stochasticDescent(X,Y,initW,0.1, 0.3, 1)
    
    y_predicted = []
    for i in range(Xtest.shape[0]):
        y_predicted.append(1 if prob_Yequal1_givenWX(newW_stochastic, Xtest[i,:]) > 0.5 else 0)
   
    printPrecisionRecall(Ytest, y_predicted)

    all_data = genfromtxt(path_prefix + 'oversampled_train.txt', delimiter=',')
    Y = all_data[:,0]
    X = add_ones(all_data[:, 1:])
    
    initW = zeros(X.shape[1])

    newW_gradient, log_loss_gradient = gradientDescent(X,Y,initW,0.01, 0.3, 10000)
    
    plt.plot(array(xrange(10001)), log_loss_gradient, 'g.--')
    plt.show()

    y_predicted = []
    for i in range(Xtest.shape[0]):
        y_predicted.append(1 if prob_Yequal1_givenWX(newW_gradient, Xtest[i,:]) > 0.5 else 0)

    printPrecisionRecall(Ytest, y_predicted)