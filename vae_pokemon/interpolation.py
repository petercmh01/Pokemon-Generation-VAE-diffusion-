import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from vae_model import VAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Load your trained model
device = 'cuda'
vae = VAE(latent_dim = 128)
vae.load_state_dict(torch.load(".../vae/vae_epoch_10000.pth")) # load your actual trained model
vae.to(device)
vae.eval()

# Load the two images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

image1 = transform(Image.open("/home/dh_01/cow_cat/vae/Pokemon/Pokemon/victini.png").convert("RGB")).unsqueeze(0).to(device)  # Replace with your own images
image2 = transform(Image.open("/home/dh_01/cow_cat/vae/Pokemon/Pokemon/zoroark.png").convert("RGB")).unsqueeze(0).to(device)  # Replace with your own images

# Encode the images into latent space
mu1, logvar1 = vae.encoder(image1)
mu2, logvar2 = vae.encoder(image2)
z1 = vae.reparameterize(mu1, logvar1)
z2 = vae.reparameterize(mu2, logvar2)

# Interpolation
num_steps = 10  # Number of intermediate images
alphas = np.linspace(0, 1, num_steps)
interpolated_images = []

for alpha in alphas:
    z_interpolated = (1 - alpha) * z1 + alpha * z2
    generated_image = vae.decoder(z_interpolated)
    interpolated_images.append(generated_image)

# Save the interpolated images
for i, img in enumerate(interpolated_images):
    save_image(img, f'.../vae/interpolated_image_{i + 1}.png')
