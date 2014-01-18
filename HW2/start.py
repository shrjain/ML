#!/usr/bin/python
from __future__ import division
from numpy import *

add_ones = lambda a: hstack((ones((a.shape[0])).reshape(a.shape[0], 1),a))
def main():
  all_data = genfromtxt('clickprediction_data/train.txt', delimiter=',')
  Y = all_data[:,0]
  X = add_ones(all_data[:, 1:])
  Xtest = add_ones(genfromtxt('clickprediction_data/test.txt', delimiter=','))
  Ytest = genfromtxt('clickprediction_data/test_label.txt')
if __name__ == '__main__':
  main()
