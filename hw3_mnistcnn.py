# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:25:34 2019

@author: shirley
"""

import torchvision
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

trainset = datasets.MNIST('./data', train = True, download = True,transform=transforms.Compose([transforms.ToTensor()]))
testset = datasets.MNIST(root='./data', train = None, download = True,transform=transforms.Compose([transforms.ToTensor()]))
train = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)#inputs have 1 channel, so they will have size 32*1*28*28
test = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

#define the CNN network
class CNNet(nn.Module):

    def __init__(self):
        super(CNNet, self).__init__()
        # 1 input image channel, 20 output channels, 3x3 square convolution
        # kernel
        self.conv = nn.Conv2d(1, 20, 3)
        self.fc1 = nn.Linear(20*13*13, 128)  
        self.fc2 = nn.Linear(128, 10)
        

    def forward(self, x): 
        # Max pooling over a (2, 2) window
        x = F.relu(F.max_pool2d(self.conv(x), (2, 2)))
        x = x.view(-1, 20*13*13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnnet = CNNet()
print(cnnet)


#define how to calculate the accuracy
def Accuracy(train_test):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in train_test:
            X, y = data
            output = cnnet(X)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
            
    Accuracy = round(correct/total, 3)
    return Accuracy

loss_function = nn.CrossEntropyLoss()

losses = []
accuracy = []
EPOCHS = 20
cnnet = CNNet()
optimizer = optim.SGD(cnnet.parameters(), lr=0.01)
for epoch in range(EPOCHS): 
    running_loss = 0.0
    for data in train:  
        X, y = data 
        cnnet.zero_grad()  
        output = cnnet(X)  
        loss = loss_function(output, y)  
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
    losses.append(running_loss)
    accuracy.append(Accuracy(train))
    print(running_loss) 
    #print(accuracy(train))

#save the model
torch.save(cnnet.state_dict(), 'mnist-cnn.pt')
#reload the model
cnnet = CNNet()
cnnet.load_state_dict(torch.load('mnist-cnn.pt'))
#report the accuracy on testset
print(Accuracy(test))

#plot loss vs epochs
plt.plot(range(20), losses, c = 'blue')
plt.xlabel('epoch')
plt.ylabel('SGDLoss')
plt.show()

#plot accuracy vs epochs
plt.plot(range(20), accuracy, c = 'red')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.show()

# use ADAM as the optimizer
lossesADAM = []
EPOCHS = 20
cnnet = CNNet()
optimizer = optim.Adam(cnnet.parameters(), lr=0.01)
for epoch in range(EPOCHS): 
    running_loss = 0.0
    for data in train: 
        X, y = data  
        cnnet.zero_grad() 
        output = cnnet(X)  
        loss = loss_function(output, y) 
        loss.backward()  
        optimizer.step()
        running_loss += loss.item()
    lossesADAM.append(running_loss)
    print(running_loss) 
    
#plot loss vs epochs using ADAM as the optimizer
plt.plot(range(20), lossesADAM, c = 'blue')
plt.xlabel('epoch')
plt.ylabel('ADAMLoss')
plt.show()

# use ADAGRAD as the optimizer
lossesADAGRAD = []
EPOCHS = 20
cnnet = CNNet()
optimizer = optim.Adagrad(cnnet.parameters(), lr=0.01)
for epoch in range(EPOCHS):
    running_loss = 0.0
    for data in train: 
        X, y = data  
        cnnet.zero_grad()  
        output = cnnet(X)  
        loss = loss_function(output, y) 
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()
    lossesADAGRAD.append(running_loss)
    print(running_loss)
    
#plot loss vs epochs using ADAGRAD as the optimizer  
plt.plot(range(20), lossesADAGRAD, c = 'green')
plt.xlabel('epoch')
plt.ylabel('ADGRADLoss')
plt.show()

#caculate convergence time
import time
time_of_conv = []
for i in [32, 64, 96, 128]:
    cnnet = CNNet()
    optimizer = optim.SGD(cnnet.parameters(), lr=0.01)
    train = torch.utils.data.DataLoader(trainset, batch_size=i, shuffle=True)#inputs have 1 channel, so they will have size 32*1*28*28
    test = torch.utils.data.DataLoader(testset, batch_size=i, shuffle=True)
    save_running_loss = [10000]
    perf_diff = 100000
    epoch_counter = 0
    start_time = time.time()
    while perf_diff > 5:
        running_loss = 0.0
        for data in train:  
                X, y = data 
                cnnet.zero_grad()  
                output = cnnet(X)  
                loss = loss_function(output, y)  
                loss.backward() 
                optimizer.step()
                running_loss += loss.item()
        save_running_loss.append(running_loss)
        perf_diff = np.abs(save_running_loss[-1]-save_running_loss[-2])
        epoch_counter += 1
    time_stop = time.time()
    time_of_conv.append(time_stop - start_time)
    
print(time_of_conv)
#plot convergence time vs batches   
plt.plot([32,64,96,128], save_time_of_conv)
plt.xlabel('batches')
plt.ylabel('covergence time')
plt.show()