# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 23:49:16 2019

@author: Xuejie Song
"""

import torchvision
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

Dataset = datasets.MNIST('./data', download = True,transform=transforms.Compose([transforms.ToTensor()]))
data = torch.utils.data.DataLoader(Dataset, batch_size=64, shuffle=True)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        

    def forward(self, x):
    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x

E = Encoder()
print(E)

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(20, 400)
        self.fc2 = nn.Linear(400, 784)
        

    def forward(self, x):
    
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x

D = Decoder()
print(D)

D_optimizer = optim.Adam(D.parameters(),lr = 0.0002)
E_optimizer = optim.Adam(E.parameters(),lr = 0.0002)
criterion = nn.BCELoss()

def images_to_vectors(images):
    return images.view(-1, 784)

def vectors_to_images(vectors):
    return vectors.view(-1, 1, 28, 28)

N_epochs = 10
Loss = []
batch_size=64
for epoch in range(N_epochs):
    train_loss = 0
    for idx_batch, (dt,_) in enumerate(data):
        
        ddt = images_to_vectors(dt)
        noise_dt = ddt + 0.5 * torch.randn(*ddt.shape)
        noise_dt = np.clip(noise_dt, 0., 1.)
        E_optimizer.zero_grad()
        Encode_data = E(noise_dt)
        D_optimizer.zero_grad()
        Dcode_data = D(Encode_data)
        loss = criterion(Dcode_data,ddt )
        loss.backward()
        E_optimizer.step()
        D_optimizer.step()
        train_loss += loss.item()
        
    train_loss = train_loss/batch_size
    Loss.append(train_loss)
    print('Epoch: {} \tLoss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    

torch.save(E.state_dict(), 'hw5_dAE_Encoder.pth')
torch.save(D.state_dict(), 'hw5_dAE_Dcoder.pth')

plt.title('dAE')
plt.plot(range(10),Loss, c = 'red',label='dAE loss')

plt.legend() 
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()

testset = datasets.MNIST(root='./data', train = None, download = True,transform=transforms.Compose([transforms.ToTensor()]))
test = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)
dataiter = iter(test)
images, idx = dataiter.next()
noisy_imgs = images + 0.5 * torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
noisy_img = images_to_vectors(noisy_imgs)
Encode_data = E(noisy_img)
D_optimizer.zero_grad()
Dcode_data = D(Encode_data)

noisy_imgs = noisy_imgs.numpy()
Dcode_data = vectors_to_images(Dcode_data).detach().numpy()
D_optimizer.step()
fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(12,7))

for noisy_imgs, row in zip([noisy_imgs, Dcode_data], axes):
    for img, ax in zip(noisy_imgs, row):
        ax.imshow(img.reshape((28,28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


