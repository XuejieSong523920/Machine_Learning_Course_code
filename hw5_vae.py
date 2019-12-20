# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 23:49:16 2019

@author: Xuejie Song
"""

import torchvision
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.autograd.variable import Variable as var
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

trainset = datasets.MNIST('./data', download = True,transform=transforms.Compose([transforms.ToTensor()]))
testset = datasets.MNIST(root='./data', train = None, download = True,transform=transforms.Compose([transforms.ToTensor()]))
train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(784, 400)
        self.mu = nn.Linear(400, 20)
        self.log_var = nn.Linear(400,20)
        

    def forward(self, x):
    
        x = F.relu(self.fc1(x))
        mean = self.mu(x)
        log_var = self.log_var(x)
        
        return mean, log_var

E = Encoder()
print(E)

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(20, 400)
        self.fc2 = nn.Linear(400, 784)
        

    def forward(self, x):
    
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        return x

D = Decoder()
print(D)

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        

    def forward(self, x):
    
        mu, log_var = self.encoder(x)

        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(mu)
       
        reconstruct_x = self.decoder(x_sample)
        return reconstruct_x, mu, log_var

print(VAE)

def sum_loss(x, new_x, mean, log_var):
    
    bce = F.binary_cross_entropy(new_x, x, size_average=False)
    KLd = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return bce + KLd

vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)
criterion = nn.BCELoss()

def images_to_vectors(images):
    return images.view(-1, 784)

def vectors_to_images(vectors):
    return vectors.reshape(-1, 1, 28, 28)

N_epochs = 10
Loss = []
batch_size=64
for epoch in range(N_epochs):
    train_loss = 0
    for idx_batch, (dt,_) in enumerate(train):
        
        ddt = images_to_vectors(dt)
        optimizer.zero_grad()
        new_dt, mu, sigma = vae(ddt)
        loss = sum_loss(ddt, new_dt, mu, sigma)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss = train_loss/batch_size
    Loss.append(train_loss)
    print('Epoch: {} \tLoss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    
torch.save(vae.state_dict(), 'hw5_vae.pth')


plt.title('VAE')
plt.plot(range(10),Loss, c = 'blue',label='VAE loss')

plt.legend() 
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()


test = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)
dataiter = iter(test)
images, labels = dataiter.next()
image = images_to_vectors(images)
new_data, mu, var = vae(image)

image = new_data.view(16,1,28,28).detach()
torch_image = torchvision.utils.make_grid(image, nrow=4)
plt.imshow(torch_image.permute(1,2,0))
plt.show()


