# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:04:27 2013

@author: shrainik
"""
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt

def clusteringKMeans(X,y,k, random = False):
    if random == False:
        centeroids = np.copy(X[0:k, :])
    else:
        randomShuffle = range(len(y))
        np.random.shuffle(randomShuffle)
        centeroid_indexes = randomShuffle[:k]
        centeroids = np.copy(X[centeroid_indexes, :])
    predictions = np.array(np.zeros(len(y)))
    for i in range(20):
        converged = True
        for j in range(len(y)):
            lowest = 0 #index of lowest euclidian distance
            for m in range(1,k):
                if spatial.distance.euclidean(centeroids[lowest], X[j]) > spatial.distance.euclidean(centeroids[m], X[j]):
                    lowest = m
            if predictions[j] != lowest:
                converged = False
            predictions[j] = lowest
            predictions[j] = int(predictions[j])
        for m in range(k):
            len_m = len(X[predictions == m])
            centeroids[m] = np.zeros(X.shape[1])
            for xi in X[predictions == m]:
                centeroids[m] += (1.0/len_m)*(xi)
    
        if converged == True:
            break
    
    return predictions, i + 1, centeroids
    
if __name__ == '__main__':
    path_prefix = '/Users/shrainik/Dropbox/ML/ML/HW4/'
    X = np.genfromtxt(path_prefix + 'digit.txt')
    Y = np.genfromtxt(path_prefix + 'labels.txt', dtype=int)
    ks = [2, 4, 6]
    for k in ks:
        predictions, iterations, centeroids = clusteringKMeans(X, Y, k)
        
        sumOfSquaresk = np.zeros(k)
        labelsPerCluster = []
        for i in range(k):
            labelsPerCluster.append({1:0, 3:0, 5:0, 7:0})
        for i in range(len(Y)):
            sumOfSquaresk[predictions[i]] += np.linalg.norm(X[i]-centeroids[predictions[i]])**2
            labelsPerCluster[int(predictions[i])][Y[i]] += 1
        
        majorityVotePerCluster = []
        mistakes = 0        
        for i in range(k):
            majorityVotePerCluster.append(labelsPerCluster[i][1])
            for j in [3,5,7]:
                if majorityVotePerCluster[i] < labelsPerCluster[i][j]:
                    majorityVotePerCluster[i] = labelsPerCluster[i][j]
            mistakes += (predictions == i).sum() - majorityVotePerCluster[i]
        
        print 'k = {0}: sum of within group sum of squares = {1} and mistake rate = {2}'.format(k, sum(sumOfSquaresk), mistakes/1000.0)
        
        print 'number of iterations for k = {0}: {1}'.format(k,iterations)
    
    ks = range(1,11)
    sums = np.zeros(len(ks))
    mistakeRates = np.zeros(len(ks))
    count = 0
    for k in ks:
        predictions, iterations, centeroids = clusteringKMeans(X, Y, k, random = True)
        sumOfSquaresk = np.zeros(k)
        labelsPerCluster = []
        for i in range(k):
            labelsPerCluster.append({1:0, 3:0, 5:0, 7:0})
        for i in range(len(Y)):
            sumOfSquaresk[predictions[i]] += np.linalg.norm(X[i]-centeroids[predictions[i]])**2
            labelsPerCluster[int(predictions[i])][Y[i]] += 1
        
        majorityVotePerCluster = []
        mistakes = 0        
        for i in range(k):
            majorityVotePerCluster.append(labelsPerCluster[i][1])
            for j in [3,5,7]:
                if majorityVotePerCluster[i] < labelsPerCluster[i][j]:
                    majorityVotePerCluster[i] = labelsPerCluster[i][j]
            mistakes += (predictions == i).sum() - majorityVotePerCluster[i]
        
        sums[count] = sum(sumOfSquaresk)
        mistakeRates[count] = mistakes/1000.0
        count +=1
    
    plt.plot(ks, sums)
    plt.xlabel('k from 1 to 10')
    plt.ylabel('Sum of within group sum of squares')
    plt.show()
    
    plt.plot(ks, mistakeRates)
    plt.xlabel('k from 1 to 10')
    plt.ylabel('MistakeRates')
    plt.show()