# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:02:34 2019

@author: Liv4D
"""
import argparse
import os
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
from models import Discriminator, Generator
from utilities import get_data_loader, compute_gradient_penalty
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--G1", type=int, default=128, help="Structure of Generator-First layer node nums")
parser.add_argument("--G2", type=int, default=256, help="Structure of Generator-Second layer node nums")
parser.add_argument("--G3", type=int, default=512, help="Structure of Generator-Third layer node nums")
parser.add_argument("--G4", type=int, default=1024, help="Structure of Generator-Forth layer node nums")
parser.add_argument("--D1", type=int, default=128, help="Structure of discriminator-First layer node nums")
parser.add_argument("--D2", type=int, default=512, help="Structure of discriminator-Second layer node nums")
parser.add_argument("--D3", type=int, default=256, help="Structure of discriminator-Third layer node nums")
parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--critic_num", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
parser.add_argument("--lambda_gp", type=int, default=10, help="lambda for gradient penalty")



setting = parser.parse_args()
print(setting)
G_structure = (setting.G1,setting.G2,setting.G3,setting.G4)
D_structure = (setting.D1,setting.D2,setting.D3)
img_shape = (setting.channels, setting.img_size, setting.img_size)


################################### Running on GPU ############################
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Weight of the Gradient Penalty
lambda_gp = 10

################################### Save path #################################
os.makedirs("images_f", exist_ok=True)
experiment_path = os.path.join("images","lr-"+str(setting.lr)+"_n_critic-"+ str(setting.critic_num) + "_G-" + str(G_structure)+ "_D-"+str(D_structure))
os.makedirs(experiment_path, exist_ok=True)





generator = Generator(setting.latent_dim, G_structure)
discriminator = Discriminator(D_structure)
generator.to(device)
discriminator.to(device)


##################################### Load Dataset ############################
trainloader, validloader, testloader = get_data_loader("svhn", setting.batch_size)
print('Download done!')


optimizer_G = torch.optim.Adam(generator.parameters(), lr=setting.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=setting.lr)

##################################### Training GAN ############################
d_loss_epochs = []
g_loss_epochs = []
batches_done = 0
for epoch in range(setting.n_epochs):
    d_loss_batches=[]
    g_loss_batches=[]
    for i, (imgs, _) in enumerate(trainloader):
        real_imgs = Variable(imgs.type(Tensor))
        optimizer_D.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], setting.latent_dim))))

        fake_imgs = generator(z)

        real_decision = discriminator(real_imgs)

        fake_decision = discriminator(fake_imgs)
  
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

        d_loss = -torch.mean(real_decision) + torch.mean(fake_decision) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        d_loss_batches.append(d_loss.item())
        if i % setting.critic_num == 0:


            fake_imgs = generator(z)

            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            
            g_loss_batches.append(g_loss.item())
            
            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]  "
                % (epoch, setting.n_epochs, i, len(trainloader), d_loss.item(), g_loss.item() )
            )

            if batches_done % setting.sample_interval == 0:
                save_image(fake_imgs.data[:36], experiment_path+"/%d.png" % batches_done, nrow=6, normalize=True)

            batches_done += setting.critic_num
    d_loss_batches_ = torch.mean(Tensor(d_loss_batches)).item()
    d_loss_epochs.append(d_loss_batches_)

    g_loss_batches_ = torch.mean(Tensor(g_loss_batches)).item()
    g_loss_epochs.append(g_loss_batches_)
    torch.save(generator,experiment_path+'/Generator%d.pt' %epoch)
 
##################################### PLOT LOSSES #############################
plt.plot(d_loss_epochs)
plt.plot(g_loss_epochs)
torch.save(generator,experiment_path+'/Generator.pt')

np.savez('losses.npz', d_loss_epochs=d_loss_epochs, g_loss_epochs=g_loss_epochs)
