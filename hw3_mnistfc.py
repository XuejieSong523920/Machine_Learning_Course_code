# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:03:44 2019

@author: Xuejie Song
"""
import torchvision
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.nn as nn

trainset = datasets.MNIST('./data', train = True, download = True,transform=transforms.Compose([transforms.ToTensor()]))
testset = datasets.MNIST(root='./data', train = None, download = True,transform=transforms.Compose([transforms.ToTensor()]))
train = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)#inputs have 1 channel, so they will have size 32*1*28*28
test = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

# define the fuuly connectted network
class fcNet(nn.Module):

    def __init__(self):
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


fcnet = fcNet()
print(fcnet)

#define how to caculate the accuracy
def Accuracy(train_test):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in train_test:
            X, y = data
            output = fcnet(X.view(-1, 28*28))
            for idx, i in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
            
    Accuracy = round(correct/total, 3)
    return Accuracy


import torch.optim as optim
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(fcnet.parameters(), lr=0.01)

losses = []
accuracy = []
EPOCHS = 20
for epoch in range(20):
    running_loss = 0.0
    for data in train: 
        X, y = data  
        fcnet.zero_grad() 
        output = fcnet(X.view(-1,784))  
        loss = loss_function(output, y)  
        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()
    print(running_loss) 
    print(Accuracy(train))
    losses.append(running_loss)
    accuracy.append(Accuracy(train))


import matplotlib.pyplot as plt
# plot loss
plt.plot(range(20), losses, c = 'blue')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()

#plot accuracy
plt.plot(range(20), accuracy, c = 'red')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.show()


#save the model
torch.save(fcnet.state_dict(), 'mnist-fc.pt') # save the model
#reload the model
fcnet = fcNet()
fcnet.load_state_dict(torch.load('./mnist-fc.pt'))
#get the accuracy of testset using this fully_connected model
print(Accuracy(test))





