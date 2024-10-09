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
import constriction

import matplotlib.pyplot as plt


from model import VAE, VAEPLUS
from trainer import train, test
import tqdm

def cal_mse(imgs, imgs_fp):
    squ_err = (imgs - imgs_fp)**2
    squ_err_sum = squ_err.sum(dim=1)
    # print(squ_err_sum.shape)
    sqroot_squ_err_sum = torch.sqrt(squ_err_sum)
    mse = sqroot_squ_err_sum.mean()
    return mse
    
def plot_bit_vs_mse(bit_vs_mse):
    plt.plot(bit_vs_mse[:, 0], bit_vs_mse[:, 1], 'o-', color='r')
    plt.xlabel('Compressed size (bits)')
    plt.ylabel('MSE')
    plt.title('Compressed size (bits) vs. MSE')
    # save the plot
    plt.savefig('./out/compression_lossy_vae_plus/bit_vs_mse.png')
    
def quantize(para_stream, qbit):
    '''
    using min-max quantization
    '''
    min_val = para_stream.min()
    max_val = para_stream.max()
    scale = (max_val - min_val) / (2**qbit - 1)
    para_stream = para_stream / scale
    para_stream = torch.clamp(para_stream, -2**(qbit-1), 2**(qbit-1) - 1)
    para_stream_int = torch.round(para_stream).int()
    return para_stream_int, scale

def dequantize(para_stream_int, scale):
    para_stream_fp = para_stream_int * scale
    return para_stream_fp

def lossy_compress(mu, log_var, qbit, device='cuda:0'):
    '''
    lossy compression using ANS encoding
    '''
    img_num = mu.shape[0]
    org_shape = mu.shape
    mu_int, mu_scale = quantize(mu, qbit)
    log_var_int, log_var_scale = quantize(log_var, qbit)
    
    mu_int = np.array(mu_int.cpu()).flatten()
    log_var_int = np.array(log_var_int.cpu()).flatten()
    
    entropy_model_mu = constriction.stream.model.QuantizedGaussian(
        mu_int.min(),
        mu_int.max(),
        mu_int.mean(),
        mu_int.std()
    )
    entropy_model_log_var = constriction.stream.model.QuantizedGaussian(
        log_var_int.min(),
        log_var_int.max(),
        log_var_int.mean(),
        log_var_int.std()
    )
    encoder_mu = constriction.stream.stack.AnsCoder()
    encoder_log_var = constriction.stream.stack.AnsCoder()
    
    encoder_mu.encode_reverse(mu_int, entropy_model_mu)
    encoder_log_var.encode_reverse(log_var_int, entropy_model_log_var)
    
    compressed_mu = encoder_mu.get_compressed()
    compressed_log_var = encoder_log_var.get_compressed()
    decoder_mu = constriction.stream.stack.AnsCoder(compressed_mu)
    decoder_log_var = constriction.stream.stack.AnsCoder(compressed_log_var)
    decoded_mu = decoder_mu.decode(entropy_model_mu, len(mu_int))
    decoded_log_var = decoder_log_var.decode(entropy_model_log_var, len(log_var_int))
    assert (mu_int == decoded_mu).all()
    assert (log_var_int == decoded_log_var).all()
    
    bit_mu_sum = sum([(len(bin(word))-2) for word in compressed_mu])
    bit_log_var_sum = sum([(len(bin(word))-2) for word in compressed_log_var])
    avg_bit_num = (bit_mu_sum + bit_log_var_sum) / img_num
    
    mu_fp = dequantize(torch.tensor(decoded_mu).to(device), mu_scale)
    log_var_fp = dequantize(torch.tensor(decoded_log_var).to(device), log_var_scale)
    return avg_bit_num, torch.reshape(mu_fp, org_shape), torch.reshape(log_var_fp, org_shape)



if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='VAE-MNIST Course Homework by Yufei')
    # model arguments
    parser.add_argument('--image-size', type=int, default=784, metavar='image-size',
                        help='input image size (default: 784)')
    parser.add_argument('--h-dim', type=int, default=400, metavar='h-dim',
                        help='hidden dimension (default: 400)')
    parser.add_argument('--latent-dim', type=int, default=20, metavar='latent-dim',
                        help='latent dimension (default: 20)')
    parser.add_argument('--z-dim', type=int, default=20, metavar='z-dim',
                        help='latent dimension (default: 20)')
    parser.add_argument('--gpu', type=str, default=0, metavar='gpu',
                        help='GPU id to use (default: 0)')
    args = parser.parse_args()
    
    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # load model
    model = VAEPLUS(args.image_size, args.h_dim, args.latent_dim, args.z_dim).to(device)
    model.load_state_dict(torch.load('./out/vae_plus/vae.pth', map_location=device))
    
    # load imags 'test_images.pth'
    img_num = 64
    img_size = (28, 28)
    imgs = torch.load('test_images.pth')  # size(64, 1, 28, 28)
    # save_image(imgs.view(64, 1, 28, 28), './out/figs/test_imgs' + '.png', nrow=8)

    with torch.no_grad():
        imgs = imgs.to(device)
        imgs = imgs.view(-1, args.image_size)
        mu, log_var = model.encode(imgs)  # size(64, 2)
        mu_shape, log_var_shape = mu.shape, log_var.shape

        qbits = [2,3,4,5,6,7,8]
        bit_vs_mse = np.zeros((len(qbits), 2))
        for qbit in qbits:
            print('qbit:', qbit)
            avg_bit_num, mu_fp, log_var_fp = lossy_compress(mu, log_var, qbit)
            z = model.reparameterization(mu_fp, log_var_fp)
            imgs_fp = model.decode(z)
            imgs_fp = imgs_fp.view(-1, args.image_size)
            mse = cal_mse(imgs, imgs_fp).cpu()
            
            bit_vs_mse[qbits.index(qbit)] = [avg_bit_num, mse]
            imgs_fp = imgs_fp.view(img_num, 1, *img_size)
            imgs_fp = torch.round(imgs_fp)
            save_image(imgs_fp, './out/compression_lossy_vae_plus/reconst_imgs_qbit' + str(qbit) + 'bit.png', nrow=8)
        plot_bit_vs_mse(bit_vs_mse)
    # a = torch.tensor([0.243242, 0.57389242, 0.4738743, 0.7483882])
    # aq, scale = quantize(a, )
    # ahat= dequantize(aq, scale)
    # print(aq)
    # print(ahat)
    
    
    
    
    