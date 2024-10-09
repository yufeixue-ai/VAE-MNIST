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

class Round:
    def __call__(self, x):
        return torch.round(x)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='VAE-MNIST Course Homework by Yufei')
    # model arguments
    parser.add_argument('--image-size', type=int, default=784, metavar='image-size',
                        help='input image size (default: 784)')
    parser.add_argument('--h-dim', type=int, default=400, metavar='h-dim',
                        help='hidden dimension (default: 400)')
    parser.add_argument('--latent-dim', type=int, default=20, metavar='z-dim',
                        help='latent dimension (default: 20)')
    parser.add_argument('--z-dim', type=int, default=2, metavar='z-dim',
                        help='latent dimension (default: 2)')
    # data arguments
    parser.add_argument('--data-dir', type=str, default='./data', metavar='data-dir',
                        help='data directory (default: ./data)')
    # training arguments
    parser.add_argument('--batch-size', type=int, default=128, metavar='batch-size',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, metavar='epochs',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', type=str, default=0, metavar='gpu',
                        help='GPU id to use (default: 0)')
    parser.add_argument('--seed', type=int, default=66, metavar='seed',
                        help='random seed ((default: 66)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='log-interval',
                        help='how many batches to wait before logging training status (default: 10)')
    # optimization arguments
    parser.add_argument('--lr', type=float, default=1e-3, metavar='lr',
                        help='learning rate (default: 1e-3)')
    args = parser.parse_args()
    
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # seed
    # torch.manual_seed(args.seed)
    # dataset
    binarizer = Round()
    train_loader = DataLoader(
        torchvision.datasets.MNIST(args.data_dir, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        torchvision.datasets.MNIST(args.data_dir, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                   ])),
        batch_size=args.batch_size, shuffle=False)   
    # model
    model = VAE(args.image_size, args.h_dim, args.latent_dim, args.z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, train_loader, optimizer, args.log_interval, device)
        test(model, epoch, test_loader, device)
        with torch.no_grad():
            if epoch % 500 == 0:
                # sample = torch.randn(64, args.z_dim).to(device)
                sample = torch.randn(64, args.z_dim).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28), './out/vae/sample_' + str(epoch) + '.png')
    
    torch.save(model.state_dict(), './out/vae/vae_org.pth')



    
