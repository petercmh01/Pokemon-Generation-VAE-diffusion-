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

## **VAE Usage**
To train vae from scratch, run
`python train_vae.py --latent_dim 128 --lr 2e-4 --epochs 10000 --batch_size 32 --dataset_path /path/to/dataset --save_path /path/to/save`

To use VAE for random generation, run
`python random_generation.py --num_images 800 --latent_dim 128 --batch_size 50 --model_path /path/to/model.pth --save_folder /path/to/save`

To interpolate between two images, run
`python interpolation.py --latent_dim 128 --model_path /path/to/model.pth --image1_path /path/to/image1.png --image2_path /path/to/image2.png --num_steps 10 --save_folder /path/to/save`

## **Diffusion**

To train diffusion model, run train_diffusion.py or follow https://github.com/lucidrains/denoising-diffusion-pytorch

## **FID**
To run FID score, run
`python -m pytorch_fid path/to/dataset1 path/to/dataset2` where one of the dataset is generated and the other is real images. 






