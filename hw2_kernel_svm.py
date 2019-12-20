# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:22:03 2019

@author: Xuejie Song
"""
from cvxopt import matrix,solvers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#part 1
def generickernelSVM(C,x, y, kernel):

    n = len(x)
    P = matrix(kernel * np.outer(y,y))
    q = matrix(-np.ones([n, 1], np.float64))
    G = matrix(np.vstack((-np.eye((n)), np.eye(n))))
    h = matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * C)))
    A = matrix(y.reshape(n,1).T)
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P,q,G,h,A,b)
    lbd = np.array(sol['x'])
    threshold = 1e-5
    S = (lbd > threshold).reshape(-1, )
    y = y.reshape(n,1)
    w = np.dot(x.T, lbd * y)
    bb = y[S] - np.dot(x[S], w)
    bb = np.mean(b)
    
    return w, bb

#part 2
pddata = pd.read_csv("hw2data.csv")
npdata = pddata.to_numpy()
N = len(npdata)
shuffled_index = np.random.permutation(N)
data_shuffled = npdata[shuffled_index]
#seperate the dataset into 80%-20%
data_imple = data_shuffled[0:int(N*0.8)+1,:]
data_test =  data_shuffled[int(N*0.8)+1:,:]
test_x = data_test[:, :-1]
test_y = data_test[:, -1].reshape(len(data_test),1)

#split the implement data into 10 folds
nFolds = 10

folds = []

nSamples = data_imple.shape[0]

nLeaveOut = nSamples // nFolds

for i in range(nFolds):
    startInd = i * nLeaveOut 
    endInd = min((i+1) * nLeaveOut, nSamples)
    
    frontPart = data_imple[:startInd, :]
    midPart = data_imple[startInd : endInd, :]
    rearPart = data_imple[endInd:, :]
    
    foldData = np.concatenate([frontPart, rearPart], axis=0)
    foldInfo={
        'train_x' : foldData[:, :-1],
        'train_y' : foldData[:, -1],
        'valid_x' : midPart[:, :-1],
        'valid_y' : midPart[:, -1]
    }
    
    folds.append(foldInfo)

def get_next_train_valid(itr):
    
    train_x = folds[itr]['train_x']
    train_y = folds[itr]['train_y']
    valid_x = folds[itr]['valid_x']
    valid_y = folds[itr]['valid_y']
    valid_y=valid_y.reshape(len(valid_y),1)
    
    return train_x, train_y, valid_x, valid_y

def rbf_svm_train(itr,C,gamma):
    
    train_x, train_y, valid_x, valid_y=get_next_train_valid(itr)
    n=len(train_x)
    x_norm = np.array(np.sum(np.multiply(train_x, train_x), axis=1)).reshape(-1,1)

    #repalce 1/2sigma^2 with gamma
    kernel = gamma * np.exp(-(x_norm + np.transpose(x_norm) - 2*np.dot(train_x, np.transpose(train_x))))
    
    kernel.reshape(len(train_x), len(train_x))
    kernel = matrix(kernel)
    kernel1 = np.ones((len(train_x),len(train_x)))
    w,b=generickernelSVM(C, train_x, train_y, kernel)
    
    return w,b

def rbf_svm_predict(x,w,bb):
    return 2*((x.dot(w)+bb)>0)-1

def validaccurary(C,gamma):
    ac = np.zeros((nFolds))
    for i in range(0,nFolds):
        train_x, train_y, valid_x, valid_y = get_next_train_valid(i)
        w, b = rbf_svm_train(i,C,gamma)
        y_predict = rbf_svm_predict(valid_x,w,b)
        valid_y = valid_y.reshape(len(valid_y),1)
        k = np.sum(np.absolute(y_predict-valid_y))/2
        n = len(valid_y)
        ac[i] = 1-k/n
        
    accu=np.mean(ac)
        
    return(accu)     

Cc = [0.0001,0.001,0.01,0.1,1,10,100,1000]
NC=len(Cc)
accu=np.zeros(NC)
for i in range(NC):
    accu[i] = validaccurary(Cc[i],0.5)
    
accu=np.array(accu)
accu=accu.reshape(NC,1)
plt.plot(Cc, accu, 'b-')
plt.show()

def testaccurary(C,gamma):
    acc = np.zeros((nFolds))
    for i in range(0,nFolds):
        train_x, train_y, valid_x, valid_y=get_next_train_valid(i)
        w,b = rbf_svm_train(i,C,gamma)
        y_predict = rbf_svm_predict(test_x,w,b)
        k = np.sum(np.absolute(y_predict-test_y))/2
        n = len(test_y)
        acc[i] = 1-k/n
    
    accu=np.mean(acc)
        
    return(accu)     
    
accut=np.zeros(NC)
for i in range(NC):
    accut[i] = validaccurary(Cc[i],0.5)
    
accut=np.array(accut)
accut=accut.reshape(NC,1)
plt.plot(Cc, accut, 'b-')
plt.show()