import torch
from torchvision.utils import save_image


G = torch.load('Generator5.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_latent = 100
img_num =1000

z = torch.randn(img_num, num_latent).to(device) 
images = G(z)

########################## Saving Images to calculate FID score ###############
for i in range(img_num):
    save_image(images.data[i], "fid_samples\samples\img_%d.png" %i, nrow=1, normalize=True)