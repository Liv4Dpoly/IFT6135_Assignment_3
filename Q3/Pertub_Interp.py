# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:37:45 2019

@author: Liv4D
"""

import torch
from torchvision.utils import save_image
import numpy as np


############################### Load your Generator ############################
G = torch.load('Generator0.pt')

################################### Running on GPU ############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_latent = 100
eps = 0.5


for eps in np.arange(0.01,1.4,1):

    def direction_pertubation(z,eps):
        num_latent = z.shape[1]
        z_space = z.repeat(num_latent+1,1)
        for dim in range(1,num_latent+1):
            z_space[dim][dim-1] = z_space[dim][dim-1]+eps
    
        images = G(z_space)*0.5+0.5
    
        save_image(images.data[1:101], "pertubation.png", nrow=10, normalize=True)
        save_image(images.data[0], "pertubation_main.png", nrow=1, normalize=True)
        return images
    def interp(z0,z1):
        z = torch.zeros(11,num_latent).to(device) 
        for a,i in enumerate(np.arange(0,1.1,0.1)):
            z[a] = i * z0 + (1-i) * z1
        img_z_avg =   G(z)*0.5+0.5
        save_image(img_z_avg.data[:11], "z_avg.png", nrow=11, normalize=True)
        
        z_new = torch.cat((z0,z1),0)
        img_new=G(z_new)*0.5+0.5
        img_0 = img_new[0]
        img_1 = img_new[1]
        
        img_avg = torch.zeros(11,3,32,32).to(device) 
        for a,i in enumerate(np.arange(0,1.1,0.1)):
            img_avg[a] = i * img_0 + (1-i) * img_1
        save_image(img_avg.data[:11], "img_avg.png", nrow=11, normalize=True)
        return img_avg
###############################################################################
        
    
    
################################### Perturbation ############################    
    z = 1*torch.randn(1, num_latent).to(device) 
    image_per = direction_pertubation(z,eps)
    #i = ([4,12,15,25,44,51,60, 62,68,93])
    save_image(image_per.data[1:101], "img%f.png" %eps, nrow=10, normalize=True)
############################### Interpolation #################################
    z0 = torch.randn(1, num_latent).to(device)
    z1 = torch.randn(1, num_latent).to(device)
    image_int = interp(z0,z1)
