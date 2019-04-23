from __future__ import print_function

import torch
import torch.utils.data
from torch.utils.data import dataset
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

###############################################################
# Configuration
dataset_location = "mnist_dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
num_epochs = 20
batch_size = 128
learning_rate = 3e-4
###############################################################
# Image binarization
def binarize(gray_img):
    bin_img = (gray_img - gray_img.min()) / gray_img.max()
    bin_img = torch.round(bin_img)
    return bin_img


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(binarize)
])
###############################################################
# Data preparation
trainvalid = datasets.MNIST(
        dataset_location, train=True,
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
        num_workers=2
    )

validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

testloader = torch.utils.data.DataLoader(
            datasets.MNIST(
            dataset_location, train=False,
            download=True, 
            transform=image_transform
        ),
        batch_size=batch_size,
    )
###############################################################
# Variational Autoencoder Class


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv_stack_encode = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ELU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5)
            )
        self.mlp_encode = nn.Sequential(
            nn.ELU(), 
            nn.Linear(256, 200)
        )
        self.mlp_decode = nn.Sequential(
            nn.Linear(100, 256),
            nn.ELU(), 
        )
        self.conv_stack_decode = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, padding=4),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=2),
            nn.Sigmoid()
            )
    #Reparameterize trick    
    def reparametrize(self, mu, logvar):
        var = logvar.exp()
        std = var.sqrt()
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, x):
        x = self.conv_stack_encode(x)
        h = self.mlp_encode(x.view(x.size(0), -1))
        mu = h[:, :100]
        logvar = h[:, 100:]
        z = self.reparametrize(mu, logvar)
        z = self.mlp_decode(z)
        x_hat = self.conv_stack_decode(z.view(z.size(0),256,1,1))
        #print(x_hat.shape)
        return x_hat, mu, logvar	

###############################################################
# Loss function
def calcul_loss(x_hat, x, mu, logvar): #average loss per mini-batch
    BCE = torch.nn.functional.binary_cross_entropy(x_hat, x.view(-1, 1,28,28), reduction='sum')
    print("BCE =",BCE)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = BCE + KLD	
    return 	loss

###############################################################
# Training function
    
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(data.view(data.size(0), 1, 28, 28))
        #print(x_hat.shape)
        #print(data.shape)
        loss = calcul_loss(x_hat, data, mu, logvar)
        loss.backward()
        train_loss += loss.item() 
        optimizer.step()
        if batch_idx % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average training loss per image: {:.4f}'.format(
          epoch, train_loss / len(trainloader.dataset)))

###############################################################
# Validation function on validation set
def validate(epoch):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(validloader):
            data = data.to(device)
            x_hat, mu, logvar = model(data)
            valid_loss += calcul_loss(x_hat, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 16)
                comparison = torch.cat([data[:n],
                                      x_hat.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/valid_reconstruction_' + str(epoch) + '.png', nrow=n)
    valid_loss /= len(validloader.dataset)   
    print('====> Average validation loss per instance (i.e. minus ELBO): {:.4f}'.format(valid_loss))


###############################################################
# Test function on test set   
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for j, (data, _) in enumerate(testloader):
            data = data.to(device)
            x_hat, mu, logvar = model(data)
            test_loss += calcul_loss(x_hat, data, mu, logvar).item()
            if j == 0:
                n = min(data.size(0), 16)
                comparison = torch.cat([data[:n],
                                      x_hat.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/test_reconstruction_' + str(epoch) + '.png', nrow=n)



    test_loss /= len(testloader.dataset)
    print('====> Average test set loss per instance (i.e. minus ELBO): {:.4f}'.format(test_loss))


###############################################################
#Importance Weight Autoencoder
K = 100    

def log_q_zx(sample, mean,logvar,dim):
    C = -torch.log(torch.FloatTensor(np.asarray([np.pi]))*2)
    return C[0]*sample.shape[dim]*0.5 - torch.sum(((sample-mean)/torch.exp(logvar*0.5))**2 + logvar,dim=dim)*0.5

def log_pz(sample,dim):

    C = -torch.log(torch.FloatTensor(np.asarray([np.pi]))*2)
    return C[0]*sample.shape[dim]*0.5 - torch.sum(sample**2, dim=dim)*0.5

def log_sum_exp_trick(log_x,dim=1):

    max_logxi = torch.max(log_x,dim=dim,keepdim=True)[0]
    return max_logxi + torch.log(torch.mean(torch.exp(log_x - max_logxi),dim=dim,keepdim=True))

def sample_z(mu,logvar):
    var = logvar.exp()
    std = var.sqrt()
    z = torch.randn_like(std).to(device)
    z = z.mul(std).add(mu)
    return z


def get_k_values(model,x):
    x_hat,mu, logvar = model(x)
    mu_k_values  = mu.repeat(K,1,1).permute(1,0,2)
    logvar_k_values = logvar.repeat(K,1,1).permute(1,0,2)
    z_k_values = sample_z(mu_k_values,logvar_k_values)
    x_hat_k_values = model.mlp_decode(z_k_values)
    x_hat_k_values = model.conv_stack_decode(x_hat_k_values.view(-1,256,1,1))
    x_hat_k_values = x_hat_k_values.view(-1,K,1,28,28)
    return x_hat_k_values, mu_k_values, logvar_k_values, z_k_values

def log_px_estimate(x_hat_k_values,x_duplicate,mu_k_values,logvar_k_values,z_k_values):
    
    eps = 1e-6
    x_hat_k_values = torch.clamp(torch.clamp(x_hat_k_values, min=eps), max=1-eps) #For stability against using sigmoid with bce
    
    bce = x_duplicate * torch.log(x_hat_k_values) + (1. - x_duplicate) * torch.log(1 - x_hat_k_values)
    print('zero values',(x_hat_k_values == 0).nonzero())
    log_p_x_z  =  torch.sum(torch.sum(torch.sum(torch.sum(bce,dim=4),dim=3),dim=2),dim=0)
    log_q_z_x = log_q_zx(z_k_values, mu_k_values,logvar_k_values,dim=2)
    log_p_z   = log_pz(z_k_values,dim=2)
    log_props    = log_p_x_z - log_q_z_x + log_p_z

    lle = torch.mean(torch.squeeze(log_sum_exp_trick(log_props,dim=1)),dim=0)
    print("lle =",lle)

    return -lle


def lle_estimate(x_hat_k_values,x,mu_k_values,logvar_k_values,z_k_values):

    N, C, iw, ih = x.shape
    x_duplicate = x.repeat(K,1,1,1,1).permute(1,0,2,3,4)
    J = log_px_estimate(x_hat_k_values,x_duplicate,mu_k_values,logvar_k_values,z_k_values)
    return J

def validation_lle(epoch):
    model.eval()
    valid_lle = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(validloader):
            data = data.to(device)
            x_hat_k_values, mu_k_values, logvar_k_values, z_k_values = get_k_values(model,data)
            
            valid_lle += lle_estimate(x_hat_k_values,data,mu_k_values,logvar_k_values,z_k_values).item()
            print("i =",i)
            print('valid_lle =',valid_lle)
    valid_lle /= len(validloader.dataset) 
    print("len validloader dataset= ", len(validloader.dataset))
    print('====> Average validation log likelihood per instance (i.e. minus ELBO): {:.4f}'.format(valid_lle))

def testset_lle(epoch):
    model.eval()
    testset_lle = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(testloader):
            data = data.to(device)
            x_hat_k_values, mu_k_values, logvar_k_values, z_k_values = get_k_values(model,data)
            
            testset_lle += lle_estimate(x_hat_k_values,data,mu_k_values,logvar_k_values,z_k_values).item()
            print("i =",i)
            print('valid_lle =',testset_lle)
    valid_lle /= len(testloader.dataset) 
    print("len testloader dataset= ", len(testloader.dataset))
    print('====> Average test set log likelihood per instance (i.e. minus ELBO): {:.4f}'.format(valid_lle))

if __name__ == "__main__":
    for epoch in range(num_epochs):
        train(epoch)
        validate(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(32, 100).to(device)  ####NOTE: consider mu and logvar from encoder ???!!! pre-Assume normal standard distribution?
            ###sample = self.reparametrize(mu, logvar)
            sample = model.mlp_decode(sample)
            sample = model.conv_stack_decode(sample.view(sample.size(0),256,1,1)).cpu()
            save_image(sample.view(-1, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
     validation_lle(epoch)
     testset_lle(epoch)
                      