# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 23:45:27 2019

@author: Xuejie Song
"""
import torchvision
import numpy as np
import torch
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

Dataset = datasets.MNIST('./data', download = True,transform=transforms.Compose([transforms.ToTensor()]))
data = torch.utils.data.DataLoader(Dataset, batch_size=100, shuffle=True)

def images_to_vectors(images):
    return images.view(-1, 784)

def vectors_to_images(vectors):
    return vectors.view(-1, 1, 28, 28)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        

    def forward(self, x):
    
        x = F.leaky_relu(self.fc1(x),0.2)
        x = F.leaky_relu(self.fc2(x),0.2)
        x = torch.sigmoid(self.fc3(x))
        
        return x

D = Discriminator()


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        

    def forward(self, x):
    
        x = F.leaky_relu(self.fc1(x),0.2)
        x = F.leaky_relu(self.fc2(x),0.2)
        x = torch.tanh (self.fc3(x))
        
        return x

G = Generator()

def noise(number):
    noise = Variable(torch.randn(number, 128))
    return noise

# first generate fake images, get real images then train D, G one after the other.
# use number of epochs as a stopping criteria
# use ADAM optimizer 
# use BCE loss
D_optimizer = optim.Adam(D.parameters(),lr = 0.0005)
G_optimizer = optim.Adam(G.parameters(),lr = 0.0005)
loss = nn.BCELoss()


def train_Discriminator(real, fake):
    
    #train o real data
    real_data_size = real.size(0)# which is also the size of fake data
    D_optimizer.zero_grad()
    pre_real = D(real)
    #for discriminator, we expect it can predict real_data as 1
    error_real = loss(pre_real, Variable(torch.ones(real_data_size,1)))
#     error_real.backward(retain_graph=True)
    error_real.backward()
    
    #train on fake data
    pre_fake = D(fake)
    #for discriminator, we expect it can predict real_data as 0
    error_fake = loss(pre_fake, Variable(torch.zeros(real_data_size,1)))
#     error_fake.backward(retain_graph=True)
    error_fake.backward()
    D_optimizer.step()
    
    return error_real+error_real


def train_Generator(fake):
    
    fake_data_size = fake.size(0)
    G_optimizer.zero_grad()
    #here if should add D_optimizer ???????????
    pre = D(fake)
    #for generator, we expect the fake_data can be predict as 1
    error = loss(pre, Variable(torch.ones(fake_data_size,1)))
#     error.backward(retain_graph=True)
    error.backward()
    G_optimizer.step()
    
    return error

N_epochs = 50
discriminator_error = []
generator_error = []
pictures = []
for epoch in range(1, N_epochs+1):
    d_error = 0
    g_error = 0
    for idx_batch, (real_data_batch,_) in enumerate(data):
        
        #N is the number of images in the batch
        N = real_data_batch.size(0)
        real_data_batch = images_to_vectors(real_data_batch)
        fake = G(noise(N))
        d_ero = train_Discriminator(real_data_batch, fake)
        fake_for_G = G(noise(N))
        g_ero = train_Generator(fake_for_G)
        d_error += d_ero
        g_error += g_ero
    print(d_error)
    print(g_error)
    discriminator_error.append(d_error)
    generator_error.append(g_error)
    if epoch % 10 == 0:
        generate_picture = G(noise(16))
        image = generate_picture.view(16,1,28,28).detach()
        torch_image = torchvision.utils.make_grid(image, nrow=4)
        plt.imshow(torch_image.permute(1,2,0))
        plt.show()
#         pictures.append(generate_picture)

torch.save(D.state_dict(), 'hw5_gan_dis.pth')
torch.save(G.state_dict(), 'hw5_gan_gen.pth')

plt.title('GAN')
plt.plot(range(50), discriminator_error, c = 'red',label='discriminator loss')
plt.plot(range(50), generator_error, c = 'blue',label='generator loss')
plt.legend() 
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()





