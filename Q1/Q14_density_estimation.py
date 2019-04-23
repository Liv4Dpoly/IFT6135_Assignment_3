#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt
import argparse
from samplers import distribution3, distribution4
from models import Discriminator

parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--samp_per_epoch", type=int, default=10000, help="mini batch size")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs")

print(3*'\n')
print("Setting:")
setting = parser.parse_args()
print(setting)


# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(30000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 200, alpha=0.5, density=1)
plt.hist(xx, 200, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))



#######--- INSERT YOUR CODE BELOW ---#######

    
D = Discriminator(1)
optimizer_D = torch.optim.Adam(D.parameters(), lr=setting.lr)

cuda_available = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




cuda_available = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cuda_available:
    D.to(device)
Tensor = torch.cuda.FloatTensor if cuda_available else torch.FloatTensor 

dist_p = iter(distribution3(setting.samp_per_epoch))
dist_q = iter(distribution4(setting.samp_per_epoch))




for epoch in range(setting.n_epochs):
    
    samples_p = next(dist_p)
    samples_q = next(dist_q)
    
    q_dist = Tensor(samples_q)
    p_dist = Tensor(samples_p)

    optimizer_D.zero_grad()

    p_decision = D(p_dist)
    q_decision = D(q_dist)
    
    d_loss = -torch.mean(torch.log(q_decision)) - torch.mean(torch.log(1 - p_decision))
    d_loss.backward()
    optimizer_D.step()
        
        

f0 = lambda x: torch.exp(-x**2/2.)/((2*np.pi)**0.5)

x = Tensor(np.linspace(-5,5,1000)).view(1000,1)
D_star = D(x)/(1 - D(x))
D_star = D_star.view(D_star.size(0))
x = x.view(1000)
f1 = torch.mul(f0(x), D_star)




############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density


x = Tensor(xx).view(1000,1)
r = D(x) # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r.cpu().detach().numpy())
plt.title(r'$D(x)$')
plt.savefig('1-4a.png')

estimate = f1.cpu().detach().numpy() # estimate the density of distribution4 (on xx) using the discriminator; 
                                    
#plt.subplot(1,2,2)
plt.figure()
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#       ncol=2, mode="expand", borderaxespad=0.)
plt.title('Estimated vs True')
plt.savefig('1-4b.png')











