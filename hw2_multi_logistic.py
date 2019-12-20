# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:31:02 2019

@author: Xuejie Song
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

pdtrain = pd.read_csv("mnist_train.csv",header=None)
train = pdtrain.to_numpy()
pdtest = pd.read_csv("mnist_test.csv",header=None)
test = pdtest.to_numpy()
N=len(train)
shuffled_index = np.random.permutation(N)
train = train[shuffled_index]

test_x=test[:,1:]
test_y=test[:,0]

#splid all test data into 100 mini-batches for the gradient descent
Nbatch=600
Lbatch=len(train)/Nbatch
batch=[]
for i in range(0,Nbatch):
    fold=train[i * int(Lbatch) :(i+1) * int(Lbatch),:]
    batch.append(fold)
    
def softmax(w, x):
    
    z = np.dot(x/255, w)
    n=len(np.sum(np.exp(z),axis=1))
    sm = np.exp(z) / np.sum(np.exp(z),axis=1).reshape(n,1)
    
    return sm

def convert(y, numClass):
    
    n = len(y)
    cy = np.zeros((n,numClass))
    for i in range(n):
        cy[i, y[i]]=1
        
    return cy

def grad(i,w, numClass):
    
    x = batch[i][:,1:]
    y = batch[i][:,0]
    y_matrix=convert(y,numClass)
    m = len(x)
    sm = softmax(w,x)
    gradient = (-1/m) * np.dot(x.T, (y_matrix - sm))
    loss = (-1/m) * np.sum(y_matrix * np.log(sm))
    
    return gradient,loss

def multi_logistic_train(learning_rate, numClass, Nepochs):
    
    numP = test_x.shape[1]
    w = np.random.randn(numP,numClass)
    cost = []
    for i in range(Nepochs):
        for j in range(Nbatch):
            gra, lossone = grad(j, w, numClass)
            w = w - (learning_rate * gra.reshape(len(gra),numClass))
            cost.append(lossone)
            
    return w, cost

def multi_logistic_predict(x):
    
    w, cost= multi_logistic_train(0.001, 10, 200)
    prob = softmax(w,x)
    pred = np.argmax(prob, axis=1)
    
    return pred, cost, w

predict_labels, loss, weights = multi_logistic_predict(test_x)

np.save("trained_weights", weights)

accuracy=np.sum(test_y==predict_labels)/float(len(test_y))
print(accuracy)

lc=np.zeros(len(loss))
for i in range(len(loss)):
    lc[i]=i
    
plt.plot(lc, loss, 'b-')
plt.show()

pd.crosstab(test_y, predict_labels, rownames = "T", colnames = "P")
