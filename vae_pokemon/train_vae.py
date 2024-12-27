import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from vae_model import VAE

def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 128
    lr = 2e-4
    epochs = 10000
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Example training loop
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(".../vae/Pokemon", transform=data_transform) # Replace with your own dataset path
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae.train()
    for epoch in range(epochs):
        train_loss = 0
        for images, _ in data_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = vae(images)
            loss = loss_function(recon_images, images, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(dataset):.4f}")

        # Save generated samples every 50 epochs
        if (epoch + 1) % 500 == 0:
            vae.eval()
            with torch.no_grad():
                z = torch.randn(25, latent_dim).to(device)
                generated_images = vae.decoder(z)
                save_image(generated_images, f'..vae/result/generated_samples_epoch_{epoch + 1}.png', nrow=5) #replace with your own save path
            vae.train()
        
            # Save the model every 50 epochs
            torch.save(vae.state_dict(), f'../vae/ckpt/vae_epoch_{epoch + 1}.pth') #replace with your own save path

