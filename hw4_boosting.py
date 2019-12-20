# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:09:24 2019

@author: Xuejie Song
"""

"""Test Error Calculator"""
import numpy as np

# true values loaded (ASSUME THIS IS HIDDEN TO YOU)
true_values = np.genfromtxt('true_values.csv', delimiter=',')
true_values = np.expand_dims(true_values, axis=1)
# sample predicted values for TA testing
# sample_preds = np.genfromtxt('sample.csv', delimiter=',')
# sample_preds = np.expand_dims(sample_preds, axis=1)
# print(sample_preds)
# print(sample_preds.shape)


def score(pred_vals):
    """Function returning the error of model
    ASSUME THIS IS HIDDEN TO YOU"""
    num_preds = len(pred_vals)
    num_true_vals = len(true_values)
    val = np.sqrt(np.sum(np.square(np.log2((true_values+1)/(pred_vals+1))))/num_true_vals)
    return round(val, ndigits=5)


# sample predicted values for TA testing
# print(score(sample_preds))

import pandas as pd
import numpy as np

mu = 4
sigma_1 = 0.2
sigma_2 = 0.2
N = len(true_values)
noise_1 = np.random.normal(mu, sigma_1, N).reshape(-1,1)
noise_2 = np.random.normal(mu, sigma_2, N).reshape(-1,1)
noise_value_1 = true_values + noise_1
noise_value_2 = true_values + noise_2

n=1
p = 0.5
size = N
vector_base = np.random.binomial(n,p,size)

def combine_noise_vector(vector):

    combine_value = np.multiply(noise_value_1.reshape(-1,1),vector.reshape(-1,1)) + np.multiply(noise_value_2.reshape(-1,1),(1 - vector).reshape(-1,1))
    return combine_value

def majority(vector):
    """
    pick the majority value some vectors whose score are smaller than the score of base_line vector
    """
    majority_vector = np.zeros(N)
    mean = np.mean(vector,axis=0)
    for i in range(len(mean)):
        if mean[i] > 0.5:
            majority_vector[i] = 1
        elif mean[i] < 0.5:
            majority_vector[i] = 0
        elif mean[i] == 0.5:
            r = np.random.rand(1)
            if r<0.5:
                majority_vector[i]=1
            else:
                majority_vector[i]=0

    return majority_vector

leave_vector = []
score_all = []
i = 0
while i<1000:
    
    new_vector = np.random.binomial(1,0.49,size)#the second parameter will effect result very significantly
    new_vector_combine = combine_noise_vector(new_vector)
    if score(new_vector_combine) < score(combine_noise_vector(vector_base)):
        leave_vector.append(new_vector)
        major_vector = majority(leave_vector).reshape(-1,1)
        score_all.append(score(major_vector))
    i += 1
    
import matplotlib.pyplot as plt


plt.plot(range(len(score_all)), score_all, c = 'blue')
plt.xlabel('boosting_time_number')
plt.ylabel('score')
plt.show()

