# **Pokemon Generation with VAE and Diffusion**

This project is developed as part of the MUST Data Science class. It provides an open-source tool for training and generating Pokemon-like images using Variational Autoencoders (VAE) and Diffusion Models.

## **Features**
- **VAE Training**: Efficiently train a Variational Autoencoder for generating compressed latent representations.
- **Diffusion Models**: Generate Pokemon images using denoising diffusion probabilistic models.

## **Installation**
Ensure you have Python installed and set up a virtual environment for the project. Then, install the required dependencies:  
`pip install torch torchvision`  

`pip install denoising_diffusion_pytorch`

`pip install pytorch-fid`

Dataset: https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types

## **Training VAE**

`python script_name.py --latent_dim 64 --lr 1e-3 --epochs 5000 --batch_size 16 --dataset_path /path/to/dataset --save_path /path/to/save`



