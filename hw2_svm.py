# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:13:29 2019

@author: Xuejie Song
"""

from cvxopt import matrix,solvers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

pddata = pd.read_csv("hw2data.csv")
npdata = pddata.to_numpy()

#shuffle the data
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
    
    """
    itr : the number of which fold will be picked as the valid data
    """
    train_x = folds[itr]['train_x']
    train_y = folds[itr]['train_y']
    valid_x = folds[itr]['valid_x']
    valid_y = folds[itr]['valid_y']
    valid_y=valid_y.reshape(len(valid_y),1)
    
    return train_x, train_y, valid_x, valid_y

def svmfit(itr,C):
    
    """
    Here, I use the cvxopt to solve the convex function of lambda then get the solve of lambda, 
    the constraints will be explained in PDF.
    """
    train_x, train_y, valid_x, valid_y=get_next_train_valid(itr)
    train_y=train_y.reshape(len(train_y),1)
    n = len(train_y)
    P = matrix(np.dot(train_x,train_x.T) * np.outer(train_y,train_y))
    q = matrix(-np.ones([n, 1], np.float64))
    G = matrix(np.vstack((-np.eye((n)), np.eye(n))))
    h = matrix(np.vstack((np.zeros((n,1)), np.ones((n,1)) * C)))
    A = matrix(train_y.reshape(n,1).T)
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P,q,G,h,A,b)
    lbd = np.array(sol['x'])
    threshold = 1e-5
    S = (lbd > threshold).reshape(-1, )
    w = np.dot(train_x.T, lbd * train_y)
    bb = train_y[S] - np.dot(train_x[S], w)
    bb = np.mean(b)
    
    return w, bb

def predict(x,w,bb):
    
    """
    This function is used to predict labels.
    """
    return 2*((x.dot(w)+bb)>0)-1

def trainaccurary(C):
    
    """
    First get the predict labels of train data and compare them with true labels, then get the accuracy.
    """
    acc=np.zeros((nFolds))
    for i in range(0,nFolds):
        
        train_x, train_y, valid_x, valid_y=get_next_train_valid(i)
        w,b=svmfit(i,C)
        y_predict=predict(train_x,w,b)
        train_y=train_y.reshape(len(train_y),1)
        k=np.sum(np.absolute(y_predict-train_y))/2
        n=len(train_y)
        acc[i]=1-k/n
        
    accurary=np.mean(acc)
        
    return accurary    

#plot average train accuracy
Cc = [0.0001,0.001,0.01,0.1,1,10,100,1000]
NC=len(Cc)
accu=np.zeros(NC)
for i in range(NC):
    accu[i] = trainaccurary(Cc[i])
    
accu=np.array(accu)
accu=accu.reshape(NC,1)
plt.plot(Cc, accu, 'b-')
plt.show()

def validaccurary(C):
    
    """
    First get the predict labels of valid data and compare them with true labels, then get the accuracy.
    """
    acc=np.zeros((nFolds))
    for i in range(0,nFolds):
        
        train_x, train_y, valid_x, valid_y=get_next_train_valid(i)
        w,b=svmfit(i,C)
        y_predict=predict(valid_x,w,b)
        valid_y=valid_y.reshape(len(valid_y),1)
        k=np.sum(np.absolute(y_predict-valid_y))/2
        n=len(valid_y)
        acc[i]=1-k/n
        
    accurary=np.mean(acc)
        
    return accurary    

#plot average valid accuracy
accuv=np.zeros(NC)
for i in range(NC):
    accuv[i] = validaccurary(Cc[i])
    
accuv=np.array(accuv)
accuv=accuv.reshape(NC,1)
plt.plot(Cc, accuv, 'b-')
plt.show()

def testaccurary(C):
    
    """
    First get the predict labels of test data and compare them with true labels, then get the accuracy.
    """
    acc=np.zeros((nFolds))
    for i in range(0,nFolds):
        
        w,b=svmfit(i,C)
        y_predict=predict(test_x,w,b)
        k=np.sum(np.absolute(y_predict-test_y))/2
        n=len(test_y)
        acc[i]=1-k/n
        
    accurary=np.mean(acc)
        
    return accurary    

#plot average test accuracy
accut=np.zeros(NC)
for i in range(NC):
    accut[i] = testaccurary(Cc[i])
    
accut=np.array(accut)
accut=accut.reshape(NC,1)
plt.plot(Cc, accut, 'b-')
plt.show()

