import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from vae_model import VAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import argparse

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="VAE Latent Space Interpolation")
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained VAE model checkpoint')
    parser.add_argument('--image1_path', type=str, required=True, help='Path to the first image')
    parser.add_argument('--image2_path', type=str, required=True, help='Path to the second image')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of intermediate images for interpolation')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save the interpolated images')

    args = parser.parse_args()

    # Load your trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = VAE(latent_dim=args.latent_dim)
    vae.load_state_dict(torch.load(args.model_path))
    vae.to(device)
    vae.eval()

    # Load the two images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image1 = transform(Image.open(args.image1_path).convert("RGB")).unsqueeze(0).to(device)
    image2 = transform(Image.open(args.image2_path).convert("RGB")).unsqueeze(0).to(device)

    # Encode the images into latent space
    mu1, logvar1 = vae.encoder(image1)
    mu2, logvar2 = vae.encoder(image2)
    z1 = vae.reparameterize(mu1, logvar1)
    z2 = vae.reparameterize(mu2, logvar2)

    # Interpolation
    alphas = np.linspace(0, 1, args.num_steps)
    interpolated_images = []

    for alpha in alphas:
        z_interpolated = (1 - alpha) * z1 + alpha * z2
        generated_image = vae.decoder(z_interpolated)
        interpolated_images.append(generated_image)

    # Save the interpolated images
    os.makedirs(args.save_folder, exist_ok=True)
    for i, img in enumerate(interpolated_images):
        save_image(img, f'{args.save_folder}/interpolated_image_{i + 1}.png')

