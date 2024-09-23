import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image  # Save a given Tensor into an image file.
from torch.utils.data import DataLoader
import numpy as np


from model import VAE
from trainer import train, test

    
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='VAE-MNIST Course Homework by Yufei')
    # model arguments
    parser.add_argument('--image-size', type=int, default=784, metavar='image-size',
                        help='input image size (default: 784)')
    parser.add_argument('--h-dim', type=int, default=400, metavar='h-dim',
                        help='hidden dimension (default: 400)')
    parser.add_argument('--z-dim', type=int, default=20, metavar='z-dim',
                        help='latent dimension (default: 20)')
    parser.add_argument('--gpu', type=str, default=0, metavar='gpu',
                        help='GPU id to use (default: 0)')
    args = parser.parse_args()
    
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    model = VAE(args.image_size, args.h_dim, args.z_dim).to(device)
    model.load_state_dict(torch.load('./out/vae.pth', map_location=device))

    scale = 5
    step = 0.4
    mean = torch.range(-scale, scale, step)
    std = torch.range(-scale, scale, step)
    num = len(mean)*len(std)
    print(len(mean))
    
    # sample = torch.randn(num, args.z_dim)
    sample = torch.randn(num, 2)
    with torch.no_grad():
        for i in range(len(mean)):
            for j in range(len(std)):
                sample[i+j*len(mean),:] = mean[i] + sample[i+j*len(mean),:] * std[j]

        # sample = torch.randn(64, args.z_dim).to(device)
        sample = sample.to(device)
        sample = model.decode(sample).cpu()
        
        # save 28*28 images with len(mean)*len(std) samples

        
        save_image(sample.view(num, 1, 28, 28), './out/figs/eval' + '.png', nrow=len(mean))
    
    
    
    
    