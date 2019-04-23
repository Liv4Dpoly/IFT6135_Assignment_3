# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:53:36 2019

@author: Liv4D
"""



import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, g_in_dim, g_out_shape):
        
        self.out_shape = g_out_shape
        self.in_dim = g_in_dim
        
        
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.in_dim, 200, normalize=False),
            *block(200, 100),
            nn.Linear(100, int(np.prod(self.out_shape))),
            nn.Tanh())

    def forward(self, z):
        out = self.model(z)
        reshaped_out = out.view(out.shape[0], self.out_shape)
        return reshaped_out

    
    
class Discriminator(nn.Module):
    def __init__(self,d_in_shape,dist='jsd'):
        self.in_shape = d_in_shape
        self.dist = dist
        super(Discriminator, self).__init__()
        if self.dist =="wd_gp":
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(self.in_shape)), 500),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(500, 250),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(250, 1),
                )
        elif self.dist == "jsd":
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(self.in_shape)), 500),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(500, 250),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(250, 1),
                nn.Sigmoid()
                )
#        self.map1 = nn.Linear(self.in_shape, 50)
#        self.map2 = nn.Linear(50, 40)
#        self.map3 = nn.Linear(40, 1)
#        self.f = nn.Sigmoid()
#        self.f1 = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, data):
        reshaped_data = data.view(data.shape[0], -1)


        decision = self.model(reshaped_data)
        return decision
#    def forward(self, x):
#        x1 = self.f1(self.map1(x))
#        x2 = self.f1(self.map2(x1))
##        x3 = self.f(self.map3(x2))
#        return x2