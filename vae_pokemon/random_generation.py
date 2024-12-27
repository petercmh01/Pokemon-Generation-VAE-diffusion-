import os
import torch
from torchvision.utils import save_image
from vae_model import VAE
import argparse

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="VAE Image Generation Script")
    parser.add_argument('--num_images', type=int, default=800, help='Number of images to generate')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for generating images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained VAE model checkpoint')
    parser.add_argument('--save_folder', type=str, required=True, help='Folder to save the generated images')

    args = parser.parse_args()

    # Parameters
    num_images = args.num_images
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    model_path = args.model_path
    save_folder = args.save_folder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim).to(device)
    vae.load_state_dict(torch.load(model_path))

    os.makedirs(save_folder, exist_ok=True)

    vae.eval()

    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            z = torch.randn(min(batch_size, num_images - i), latent_dim).to(device)
            generated_images = vae.decoder(z)
            for j, img in enumerate(generated_images):
                save_path = os.path.join(save_folder, f"image_{i + j + 1:04d}.png")
                save_image(img, save_path)

    print(f"Generated {num_images} images and saved to {save_folder}.")
