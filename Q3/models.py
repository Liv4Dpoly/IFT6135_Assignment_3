# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:50:06 2019

@author: Liv4D
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch


#class Generator(nn.Module):
#    def __init__(self,latent_dim):
#        super(Generator, self).__init__()
#        self.latent_dim = latent_dim
#        
#        self.linear_1 = nn.Linear(latent_dim, 64*3*3)
#        
#               
#        self.deconv_1 = nn.ConvTranspose2d(in_channels=64,
#                                                     out_channels=32,
#                                                     kernel_size=(2, 2),
#                                                     stride=(2, 2),
#                                                     padding=0)
#                                 
#        self.deconv_2 = nn.ConvTranspose2d(in_channels=32,
#                                                     out_channels=16,
#                                                     kernel_size=(4, 4),
#                                                     stride=(2, 2),
#                                                     padding=0)
#        
#        self.deconv_3 = nn.ConvTranspose2d(in_channels=16,
#                                                     out_channels=3,
#                                                     kernel_size=(6, 6),
#                                                     stride=(2, 2),
#                                                     padding=0)
#
#
#
#    def forward(self, features):
#        
#        x = self.linear_1(features)
#        x = x.view(-1, 64, 3, 3)
#        
#        x = self.deconv_1(x)
#        x = F.leaky_relu(x)
#        #print('deconv1 out:', x.size())
#        
#        x = self.deconv_2(x)
#        x = F.leaky_relu(x)
#        #print('deconv2 out:', x.size())
#        
#        x = self.deconv_3(x)
#        x = F.leaky_relu(x)
#        #print('deconv1 out:', x.size())
#        
#        decoded = torch.tanh(x)
#        
#        return decoded




################################################################################
class Generator(nn.Module):
    def __init__(self, g_in_dim,G_structure):
        
        self.img_shape = (3,32,32)
        self.in_dim = g_in_dim
        (self.unit1, self.unit2, self.unit3, self.unit4) = G_structure
        
        super(Generator, self).__init__()    
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.in_dim, self.unit1, normalize=False),
            *block(self.unit1, self.unit2),
            *block(self.unit2, self.unit3),
            *block(self.unit3, self.unit4),
            nn.Linear(self.unit4, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img_shape = self.img_shape
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

################################################################################
#class Discriminator(nn.Module):
#    def __init__(self):
#        super(Discriminator, self).__init__()
#        self.conv_1 = nn.Conv2d(in_channels=3,
#                                          out_channels=16,
#                                          kernel_size=(6, 6),
#                                          stride=(2, 2),
#                                          padding=0)
#
#        self.conv_2 = nn.Conv2d(in_channels=16,
#                                          out_channels=32,
#                                          kernel_size=(4, 4),
#                                          stride=(2, 2),
#                                          padding=0)                 
#        
#        self.conv_3 = nn.Conv2d(in_channels=32,
#                                          out_channels=64,
#                                          kernel_size=(2, 2),
#                                          stride=(2, 2),
#                                          padding=0)                     
#        
#        self.z = torch.nn.Linear(64*3*3, 1)
#
#        
#    def forward(self, features):
#        x = self.conv_1(features)
#        x = F.leaky_relu(x)
#        #print('conv1 out:', x.size())
#        
#        x = self.conv_2(x)
#        x = F.leaky_relu(x)
#        #print('conv2 out:', x.size())
#        
#        x = self.conv_3(x)
#        x = F.leaky_relu(x)
#        #print('conv3 out:', x.size())
#        
#        out = self.z(x.view(-1, 64*3*3))
#
##        z_log_var = self.z_log_var(x.view(-1, 64*3*3))
#        
#        return out
################################################################################    
class Discriminator(nn.Module):
    def __init__(self,D_structure):      
        super(Discriminator, self).__init__()
        self.img_shape =(3, 32, 32)
        (self.unit1, self.unit2, self.unit3) = D_structure
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), self.unit1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.unit1, self.unit2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.unit2, self.unit3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.unit3, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

