# -*- coding: utf-8 -*-

#!/usr/bin/env python
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
from torch import nn
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
import os
from torchvision.utils import save_image

#Settings
if not os.path.exists('./results'):
    os.mkdir('./results')
if not os.path.exists('./results/interpolation'):
    os.mkdir('./results/interpolation')
if not os.path.exists('./results/pertubation'):
    os.mkdir('./results/pertubation')
if not os.path.exists('./fid_samples'):
    os.mkdir('./fid_samples')    
if not os.path.exists('./fid_samples/samples'):
    os.mkdir('./fid_samples/samples')
    
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Hyperparameters
random_seed = 0
learning_rate = 0.001
num_epochs = 25
batch_size = 32

# Architecture
num_features = 3*32*32
num_latent = 100



image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])


def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader

trainloader, validloader, testloader = get_data_loader("svhn", batch_size)

class VAE(torch.nn.Module):

    def __init__(self, num_features, num_latent):
        super(VAE, self).__init__()
        
        # Encoder
        self.conv1_encoder = torch.nn.Conv2d(in_channels=3,
                                          out_channels=32,
                                          kernel_size=(6, 6),
                                          stride=(2, 2),
                                          padding=0)

        self.conv2_encoder = torch.nn.Conv2d(in_channels=32,
                                          out_channels=64,
                                          kernel_size=(4, 4),
                                          stride=(2, 2),
                                          padding=0)                 
        
        self.conv3_encoder = torch.nn.Conv2d(in_channels=64,
                                          out_channels=128,
                                          kernel_size=(2, 2),
                                          stride=(2, 2),
                                          padding=0)                     
        
        self.z_mean = torch.nn.Linear(128*3*3, num_latent)
        self.z_log_var = torch.nn.Linear(128*3*3, num_latent)

        # Decoder, the same as generator of GAN
        self.img_shape = (3,32,32)
        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.decode = torch.nn.Sequential(
            *block(num_latent, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, int(np.prod(self.img_shape))),
            torch.nn.Tanh())

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def encoder(self, imgs):
        x = self.conv1_encoder(imgs)
        x = F.leaky_relu(x)      
        x = self.conv2_encoder(x)
        x = F.leaky_relu(x)        
        x = self.conv3_encoder(x)
        x = F.leaky_relu(x)
        
        z_mean = self.z_mean(x.view(-1, 128*3*3))
        z_log_var = self.z_log_var(x.view(-1, 128*3*3))
        encoded = self.reparameterize(z_mean, z_log_var)
        
        return z_mean, z_log_var, encoded
    
    def decoder(self, encoded):        
        decoded = self.decode(encoded)
        decoded = decoded.view(-1,3,32,32)
        return decoded

    def forward(self, imgs):        
        z_mean, z_log_var, encoded = self.encoder(imgs)
        decoded = self.decoder(encoded)
        
        return z_mean, z_log_var, encoded, decoded

    
torch.manual_seed(random_seed)
model = VAE(num_features, num_latent)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(epoch):
    model.train()
    train_loss=0


    for batch_idx, (imgs, targets) in enumerate(trainloader):

        imgs = imgs.to(device)
        z_mean, z_log_var, encoded, decoded = model(imgs)
        KLD = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        MSE = F.mse_loss(decoded, imgs, reduction='sum')
        loss = MSE + KLD
        train_loss +=loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not batch_idx % 50:
            print ('Epoch: %02d| Batch %04d | loss: %.2f' 
                   %(epoch+1, batch_idx, loss))
    
    train_loss/=len(trainloader.dataset)
    print('Train loss: {:.2f}'.format(train_loss))

def validate(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(validloader):
            imgs = imgs.to(device)
            z_mean, z_log_var, encoded, decoded = model(imgs)
            KLD = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
            MSE = F.mse_loss(decoded, imgs, reduction='sum')
            val_loss += (KLD + MSE)
            if batch_idx == 0:
                n_images = 4
                orig_images = imgs[:n_images]
                orig_images = orig_images*0.5+0.5 #inverse normalization
                decoded_images = decoded.view(-1, 3, 32, 32)[:n_images]
                decoded_images = decoded_images*0.5+0.5 #inverse normalization
                comparison = torch.cat([orig_images, decoded_images])
                save_image(comparison.cpu(), 'results/comparison_' + str(epoch) + '.png', nrow=n_images)
        
        val_loss/=len(validloader.dataset)
        print('Validation loss: {:.2f}'.format(val_loss))

def generation_with_interpolation(epoch,num_latent,index):
    z_0 = torch.randn(batch_size, num_latent).to(device)
    z_1 = torch.randn(batch_size, num_latent).to(device)
    x_hat_images =[]
    x_prime_images = []
    for alpha in torch.arange(0.0, 1.1, 0.1):    
        z_prime = alpha*z_0 + (1-alpha)*z_1
        x_primes = model.decoder(z_prime)
        x_prime = x_primes[0]
        x_prime = x_prime*0.5+0.5 #inverse normalization
        x_prime_images.append(x_prime)
        x_0 = model.decoder(z_0)
        x_0 = x_0[0]
        x_0 = x_0*0.5+0.5 #inverse normalization
        x_1_s = model.decoder(z_1)
        x_1 = x_1_s[0]
        x_1 = x_1*0.5+0.5 #inverse normalization
        x_hat = alpha*x_0 + (1-alpha)*x_1
        x_hat_images.append(x_hat)
        
    
    save_image(x_prime_images, 'results/interpolation/latent_interpolated_imgs_'+ str(epoch) + '_'+ str(index)+'.png', nrow=11)
    save_image(x_hat_images, 'results/interpolation/output_interpolated_imgs_' + str(epoch) + '_'+ str(index)+ '.png', nrow=11)

def generation_with_direction_pertubation(epoch,eps,num_latent,index):
    pertu_images = []    
    for dim in range(num_latent):
        z = torch.randn(batch_size, num_latent).to(device)  
        z[:,dim] = z[:,dim]+eps
        x_hat_s = model.decoder(z)
        x_hat = x_hat_s[0]
        x_hat = x_hat*0.5+0.5  #inverse normalization
        pertu_images.append(x_hat)
    save_image(pertu_images, 'results/pertubation/pertu_imgs_' + '_'+ str(epoch) + '_'+ str(index)+ '.png', nrow=10)


if __name__ == "__main__":
    n_images = 5
    eps = 5
    for epoch in range(num_epochs):
        train(epoch)
        validate(epoch)
        
    for i in range (n_images):    
        generation_with_interpolation(num_epochs,num_latent,i)
        generation_with_direction_pertubation(num_epochs,eps,num_latent,i)

    # Random sampling      
    n_images = 30
    rand_imgs = torch.randn(n_images, num_latent).to(device)
    decoded_images = model.decoder(rand_imgs)
    generated_images = decoded_images*0.5+0.5 #inverse normalization
    save_image(generated_images.view(-1, 3, 32, 32),
                   './results/generated_images_{}.png'.format(epoch), nrow=15)      
    
    # Random sampling for FID score calculation                
    n_images = 1000
    iter_times = 40
    iter_batch = 25
    for i in range(iter_times):
        rand_imgs = torch.randn(iter_batch, num_latent).to(device)
        decoded_images = model.decoder(rand_imgs)
        generated_images = decoded_images*0.5+0.5 #inverse normalization
        for j in range(iter_batch):
            index = i*iter_batch+j
            save_image(generated_images[j].view(-1, 3, 32, 32),
                   './fid_samples/samples/generated_images_{}.png'.format(index))  