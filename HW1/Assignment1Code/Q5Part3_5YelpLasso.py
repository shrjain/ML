# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 02:12:18 2013

@author: shrainik
This solution uses sparse matrices
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.io as io
import time as time
from operator import itemgetter

def lasso_solver(X,wi,w0i,y,lmbd):
    convergence = False
    delta = 0.2
    currentIteration = 0
    maxIterations = 3000
    n,d = X.shape

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
            if sparse.issparse(X):
                aj = 2*((X.T[j].todense().getA()**2).sum())
                jth_col = np.reshape(X[:,j].todense().getA(),[n,1])
            else:
                aj = 2*((X.T[j]**2).sum())
                jth_col = np.reshape(X[:,j],[n,1])
            
            cj = np.dot(jth_col.T,y - w0 - X.dot(w_dum))
            if	cj < lmbd * -1:
                wj_hat = (cj+lmbd)/aj
            elif cj > lmbd:
                wj_hat = (cj-lmbd)/aj
            else:
                wj_hat = 0
            w[j] = wj_hat
        w0 = (1.0/len(y))*(sum(y*1.0-(X.dot(w).getA())))
        convergence = True
        for i in range(d):
            if np.absolute(w[i] - w_old[i])/w_old[i] > delta:
                convergence = False 
                break
        if convergence == True:
            if np.absolute(w0_old - w0)/w0_old > delta:
                convergence = False
    return w0,w

def RMSE(y,y_actual):
    return np.sqrt(np.sum(np.square(y-y_actual))/len(y))

if __name__ == "__main__":
    y = np.loadtxt("C:\Users\shrjain\Downloads\data\star_labels.txt", dtype=np.int)
    # Load a text file of strings:
    featureNames = open("C:\Users\shrjain\Downloads\data\star_features.txt").read().splitlines()
    # Load a matrix market matrix, convert it to csc format:
    A = io.mmread("C:\Users\shrjain\Downloads\data\star_data.mtx").tocsc()
    
    #A = A.todense().getA()
    A_train = A[:30000]
    A_validation = A[30001:37501]
    A_test = A[37500:]
    y_train = np.matrix(y[:30000]).T
    y_validation = np.matrix(y[30001:37501]).T
    y_test = np.matrix(y[37500:]).T
    factor = 1.3

    print 'All data loaded.. starting now...'
    start = time.time()
    lambda_max = 2*max(np.absolute(A_train.transpose().dot(y_train - np.average(y_train))))[0,0]
    vldtnError = []
    trainError = []
    lambda_array = []
    numNonZeroes = []
    w0store = []
    lambda_array.append(lambda_max)
    w_int = np.matrix(np.zeros(2500)).T
    w_hat1 = lasso_solver(A_train,w_int,0,y_train,lambda_max)
    W_HAT = w_hat1[1].T
    w0store.append(w_hat1[0])
    vldtnError.append(RMSE(A_validation.dot(w_hat1[1])+w_hat1[0],y_validation))
    trainError.append(RMSE(A_train.dot(w_hat1[1])+w_hat1[0],y_train))
    numNonZeroes.append(sum(W_HAT!=0))
    
    lamda = lambda_max/factor
    iterationCount = 1
    condition = True
    while condition == True and iterationCount < 10000:
        print 'iteration ', iterationCount
        lambda_array.append(lamda)
        w_hat1 = lasso_solver(A_train,w_hat1[1],w_hat1[0],y_train,lamda)
        W_HAT = np.vstack((W_HAT, w_hat1[1].T))
        w0store.append(w_hat1[0])
        vldtnError.append(RMSE(A_validation.dot(w_hat1[1])+w_hat1[0],y_validation))
        trainError.append(RMSE(A_train.dot(w_hat1[1])+w_hat1[0],y_train))
        numNonZeroes.append(sum(w_hat1[1]!=0))
        if vldtnError[iterationCount] > vldtnError[iterationCount - 1]:
            if iterationCount > 5:
                condition = False
        lamda = lamda/factor
        iterationCount += 1
    end = time.time()
    print end - start
    
    plt.plot(lambda_array, vldtnError, 'r.--')
    plt.plot(lambda_array, trainError, 'g.--')
    plt.xlabel('Lambda')
    plt.legend(('vldtnError','trainError'), loc='best')
    plt.show()
    plt.plot(lambda_array, numNonZeroes, 'b.--')
    plt.xlabel('Lambda')
    plt.ylabel('numNonZeroes')
    plt.show()
    print vldtnError, trainError, lambda_array, numNonZeroes
    validFeatures = []
    wForValidFeatures = []
    bestW = 9999 if iterationCount == 10000 else iterationCount - 2
    print 'lambda for best w', lambda_array[bestW]
    print 'RMSE for best w on test data', RMSE(A_test.dot(W_HAT[bestW][0,:].T)+w0store[bestW], y_test)
    absWeights = []
    for i in range(2500):
        if W_HAT[bestW][0,i] !=0:
            validFeatures.append(featureNames[i])
            wForValidFeatures.append(W_HAT[bestW][0,i])
            absWeights.append(np.abs(W_HAT[bestW][0,i]))
    print sorted(zip(validFeatures, absWeights, wForValidFeatures), key = itemgetter(1), reverse= True)[:10]