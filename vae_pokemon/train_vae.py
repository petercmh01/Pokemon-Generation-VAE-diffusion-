import argparse
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
    # Argument parser
    parser = argparse.ArgumentParser(description="VAE Training Script")
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--save_path', type=str, default='../vae', help='Path to save models and results')

    args = parser.parse_args()

    # Hyperparameters
    latent_dim = args.latent_dim
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Dataset and DataLoader
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(args.dataset_path, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
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

        # Save generated samples every 500 epochs
        if (epoch + 1) % 500 == 0:
            vae.eval()
            with torch.no_grad():
                z = torch.randn(25, latent_dim).to(device)
                generated_images = vae.decoder(z)
                save_image(generated_images, f'{args.save_path}/result/generated_samples_epoch_{epoch + 1}.png', nrow=5)
            vae.train()

            # Save the model every 500 epochs
            torch.save(vae.state_dict(), f'{args.save_path}/ckpt/vae_epoch_{epoch + 1}.pth')
