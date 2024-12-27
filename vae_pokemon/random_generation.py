import os
import torch
from torchvision.utils import save_image
from vae_model import VAE

num_images = 800 # replace with the number of images you want to generate
latent_dim = 128  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim).to(device)  # 
vae.load_state_dict(torch.load(".../vae/ckpt4/vae_epoch_10000.pth"))

save_folder = ".../vae/generated_images" # replace with the folder where you want to save the generated images

os.makedirs(save_folder, exist_ok=True)

vae.eval()

batch_size = 50 #batch size for generating images

with torch.no_grad():
    for i in range(0, num_images, batch_size):
     
        z = torch.randn(min(batch_size, num_images - i), latent_dim).to(device)
        
        generated_images = vae.decoder(z)
      
        for j, img in enumerate(generated_images):
            save_path = os.path.join(save_folder, f"image_{i + j + 1:04d}.png")
            save_image(img, save_path)

print(f"Generated {num_images} images and saved to {save_folder}.")