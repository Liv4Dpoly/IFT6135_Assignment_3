# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:02:34 2019

@author: Liv4D
"""
from models import Discriminator
from samplers import distribution1, distribution2, distribution3, distribution4

import torch

from torch.autograd import Variable
from utilities import compute_gradient_penalty

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
parser = argparse.ArgumentParser()

parser.add_argument("--distance", type=str, default="jsd", help="jsd or wd_gp")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for SGD optimizer")
parser.add_argument("--batch_size", type=int, default=1024, help="mini batch size")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--latent_dim", type=int, default=4, help="dimensionality of the latent space")
parser.add_argument("--lambda_gp", type=int, default=10, help="lambda for gradient penalty")
parser.add_argument("--n_critic", type=int, default=2, help="number of training steps for discriminator per iter")


sample_num = 10000


print(6*'\n')
print("Setting:")
setting = parser.parse_args()
print(setting)

################################### Sampling ##################################
xx = torch.randn(sample_num)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
###############################################################################

############################## Building the nets ##############################
D = Discriminator(2,setting.distance)
###############################################################################


############################## Optimizer definung #############################
optimizer_D = torch.optim.Adam(D.parameters(), lr=setting.lr)
###############################################################################


################################ Running on GPU ###############################
cuda_available = True if torch.cuda.is_available() else False
#cuda_available = False
if cuda_available:
    D.cuda()
Tensor = torch.cuda.FloatTensor if cuda_available else torch.FloatTensor
###############################################################################


################################ Distribution p ###############################
dist_p = iter(distribution1(0,sample_num))
samples = next(dist_p)
samples_p = Tensor(samples)

D_LOSSES=[]
for theta in np.arange(-1.,1.1,0.1):
    dist_q = iter(distribution1(theta,sample_num))
    samples = next(dist_q)
    samples_q = Tensor(samples)
    batches_done = 0
    wsd = []
    for epoch in range(setting.n_epochs):
        fakes = []
        for i in range(0,sample_num,setting.batch_size):
            up_bnd = i + setting.batch_size
            if up_bnd > sample_num + 1:
                up_bnd = sample_num + 1
            real_data = samples_p[i:up_bnd]
            real_data = Variable(real_data.view(real_data.shape[0],2))
            
            fake_data = samples_q[i:up_bnd]
            fake_data = Variable(fake_data.view(fake_data.shape[0],2))
                    

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
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Teta = %f]" % (epoch, setting.n_epochs, i,sample_num , d_loss.item(),theta))
        wsd.append(d_loss.item()) 
    D_LOSSES.append(-d_loss)
               
plt.plot(np.arange(-1.,1.1,0.1),D_LOSSES,'ro')
plt.plot(np.arange(-1.,1.1,0.1),D_LOSSES,'c--')
plt.title("Distributions Distances")
plt.xlabel(r'$\Phi$')
if setting.distance == "wd_gp":
    plt.ylabel("Wasserstein Distance (WD)")
    plt.savefig("wsd-1-3.png")
else:
    plt.ylabel("Jensen Shannon Divergence (JSD)")
    plt.savefig("jsd-1-3.png")


plt.show()