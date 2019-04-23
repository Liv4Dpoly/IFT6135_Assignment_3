# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:18:54 2019

@author: Liv4D
"""

from models import Discriminator
from samplers import distribution1, distribution2, distribution3, distribution4

import torch

from torch.autograd import Variable

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from utilities import compute_gradient_penalty

start = time.time()
parser = argparse.ArgumentParser()

parser.add_argument("--distance", type=str, default="wd_gp", help="jsd or wd_gp")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for SGD optimizer")
parser.add_argument("--sample_num", type=int, default=1000, help="mini batch size")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--latent_dim", type=int, default=4, help="dimensionality of the latent space")
parser.add_argument("--lambda_gp", type=int, default=10, help="lambda for gradient penalty")
parser.add_argument("--n_critic", type=int, default=2, help="number of training steps for discriminator per iter")



print(6*'\n')
print("Setting:")
setting = parser.parse_args()
print(setting)

############################## Building the nets ##############################
#G = Generator(setting.latent_dim,1)
D = Discriminator(1,setting.distance)
###############################################################################


############################## Optimizer definung #############################
#optimizer_D = torch.optim.SGD(D.parameters(), lr=setting.lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=setting.lr)
###############################################################################


################################ Running on GPU ###############################
cuda_available = True if torch.cuda.is_available() else False
#cuda_available = False
if cuda_available:
#    G.cuda()
    D.cuda()
Tensor = torch.cuda.FloatTensor if cuda_available else torch.FloatTensor
###############################################################################


################################### Sampling ##################################
dist_r = iter(distribution4(setting.sample_num))
dist_f = iter(distribution2(setting.sample_num))
###############################################################################

################################### Training ##################################

wsd = []
for epoch in range(setting.n_epochs):

    real_loader_ = next(dist_r)
    real_data = Tensor(real_loader_)
    real_data = Variable(real_data.view(real_data.shape[0],1))
    

    fake_loader_ = next(dist_f)
    fake_data = Tensor(fake_loader_)
    fake_data = Variable(fake_data.view(fake_data.shape[0],1))
                
        
    optimizer_D.zero_grad()


    if setting.distance == "wd_gp":
        real_decision = D(real_data)

        fake_decision = D(fake_data)
        gradient_penalty = compute_gradient_penalty(D, real_data.data, fake_data.data)

        d_loss = -torch.mean(real_decision) + torch.mean(fake_decision) + setting.lambda_gp * gradient_penalty
    elif setting.distance == "jsd":
        real_decision = D(real_data)

        fake_decision = D(fake_data)
        d_loss = -np.log(2)- 0.5 * torch.mean(torch.log_(real_decision+1-1)) - 0.5 * torch.mean(torch.log_(1 - fake_decision+1-1))
    d_loss.backward()
    optimizer_D.step()



    print("[Epoch %d/%d] [D loss: %f]" % (epoch, setting.n_epochs , -d_loss.item()))
    wsd.append(d_loss.item())    
###############################################################################
    
    
################################## Plot images ################################
plt.plot(range(1,1+setting.n_epochs),wsd)
plt.title("Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("D Loss")
if setting.distance =='jsd':
    plt.savefig("1-1-jsd.png")
else:
    plt.savefig("1-2-wsd.png")
plt.show()

end = time.time()
print("Total time: %fs" % (end-start))