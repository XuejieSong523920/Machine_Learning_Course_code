# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:00:32 2019

@author: Xuejie Song
"""

import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt 

Mushroom = pd.read_csv("Mushroom.csv", header = None)

train = Mushroom.iloc[0:6000,:]
features_train = train.iloc[:,1:]
target_train = train.iloc[:,0]
test = Mushroom.iloc[6001:,:]
features_test = test.iloc[:,1:]
target_test = test.iloc[:,0]

def r_f(num_sample, feature_num):
    
    """
    num_sample: int /the number of samples I use
    feature_num: int /max_features
    """
    decision_tree_sample = []
    for i in range(num_sample):
        train_sample = train.sample(n=len(train), frac=None, replace=True, random_state=None, axis=0)
        target_sample = train_sample.iloc[:,0]
        features_sample = train_sample.iloc[:,1:]
        decision_tree = tree.DecisionTreeClassifier(criterion='gini',max_depth=2, max_features=feature_num)
        decision_tree_model = decision_tree.fit(features_sample, target_sample)
        decision_tree_sample.append(decision_tree_model)
        
    return decision_tree_sample

def ensemble_result(num_sample, feature_num, features):
    """
    features: which can come from train set or test set
    """
    decision_tree_sample = r_f(num_sample, feature_num)
    
    decision_tree_predict = []
    for i in range(num_sample):
        decision_tree_predict.append(decision_tree_sample[i].predict(features))
    
    predict_result = np.zeros(len(decision_tree_predict[0]))
    for i in range(len(decision_tree_predict[0])):
        positive_num = 0
        negative_num = 0
        for j in range(len(decision_tree_predict)):
            if decision_tree_predict[j][i]==-1:
                negative_num += 1
            else:
                positive_num += 1

        if positive_num > negative_num:
            predict_result[i]=1
        elif positive_num < negative_num:
            predict_result[i]=-1

        else:
            r = np.random.rand(1)
            if r<0.5:
                predict_result[i]=1
            else:
                predict_result[i]=-1

    return predict_result


def error_rate(num_sample, feature_num, features, target):
    
    predict = ensemble_result(num_sample, feature_num, features)
    
    return sum(predict == target)/len(target)

import matplotlib.pyplot as plt

train_accuracy = []
for i in [5,10,15,20]:
    train_accuracy.append(error_rate(100, i, features_train, target_train))

plt.plot([5,10,15,20], train_accuracy, c = 'blue')
plt.xlabel('feature set size')
plt.ylabel('train set accuracy')
plt.show()

test_accuracy = []
for i in [5,10,15,20]:
    test_accuracy.append(error_rate(100, i, features_test, target_test))

plt.plot([5,10,15,20], test_accuracy, c = 'blue')
plt.xlabel('feature set size')
plt.ylabel('test set accuracy')
plt.show()

#it is just a stright line, which might because that the number of features chose are so many!
train_accuracy = []
for i in [10,20,40,80,100]:
    train_accuracy.append(error_rate(i, 20, features_train, target_train))

plt.plot([10,20,40,80,100], train_accuracy, c = 'blue')
plt.xlabel('number of decision trees')
plt.ylabel('train set accuracy')
plt.show()


test_accuracy = []
for i in [10,20,40,80,100]:
    test_accuracy.append(error_rate(i, 20, features_test, target_test))

plt.plot([10,20,40,80,100], test_accuracy, c = 'blue')
plt.xlabel('number of decision trees')
plt.ylabel('test set accuracy')
plt.show()

