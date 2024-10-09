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


from model import VAE
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
    plt.savefig('./out/compression_lossy_vae/bit_vs_mse.png')
    plt.close()
    
def plot_qbit_vs_bit(qbit_vs_bit):
    plt.plot(qbit_vs_bit[:, 0], qbit_vs_bit[:, 1], 'o-', color='r')
    plt.xlabel('Quantization bits')
    plt.ylabel('Compressed size (bits)')
    plt.title('Quantization bits vs. Compressed size (bits)')
    # save the plot
    plt.savefig('./out/compression_lossy_vae/qbit_vs_bit.png')
    plt.close()
    
def quantize(para_stream, qbit):
    '''
    using min-max quantization
    '''
    if qbit>0:
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

def fake_quantize(para_stream, qbit):
    '''
    using min-max quantization
    '''
    if qbit>0:
        min_val = para_stream.min()
        max_val = para_stream.max()
        scale = (max_val - min_val) / (2**qbit - 1)
        para_stream = para_stream / scale
        para_stream = torch.clamp(para_stream, -2**(qbit-1), 2**(qbit-1) - 1)
        para_stream_int = torch.round(para_stream).int()
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

def lossless_compress(mu, log_var, qbit, x, x_prob, device='cuda:0'):
    '''
    lossy compression using ANS encoding
    '''
    img_num = mu.shape[0]
    org_shape = mu.shape
    mu_int, mu_scale = quantize(mu, qbit)
    log_var_int, log_var_scale = quantize(log_var, qbit)
    
    mu_int = np.array(mu_int.cpu()).flatten()
    log_var_int = np.array(log_var_int.cpu()).flatten()
    x = np.array(x.cpu()).flatten().astype(np.int32)
    x_prob = np.array(x_prob.cpu()).flatten()
    
    print('x:', x)
    print('x_prob:', x_prob)
    
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
    
    # compress binary image x
    entropy_model_x = constriction.stream.model.Bernoulli()
    encoder_x = constriction.stream.stack.AnsCoder()
    encoder_x.encode_reverse(x, entropy_model_x, np.zeros_like(x_prob)+0.5)
    compressed_x = encoder_x.get_compressed()
    
    bit_mu_sum = sum([(len(bin(word))-2) for word in compressed_mu])
    bit_log_var_sum = sum([(len(bin(word))-2) for word in compressed_log_var])
    bit_x_sum = sum([(len(bin(word))-2) for word in compressed_x])
    avg_bit_num = (bit_mu_sum + bit_log_var_sum + bit_x_sum) / img_num
    return avg_bit_num


# def lossy_compress(Z, MU, LOG_VAR, qbit, device='cuda:0'):
#     img_num = Z.shape[0]    
#     bit_sum = 0
#     for i in range(img_num):
#         z = Z[i,:]
#         mu = MU[i,:]
#         log_var = LOG_VAR[i,:]
        
#         z_int, z_scale = quantize(z, qbit)
#         z_int = np.array(z_int.cpu()).flatten()
#         mu = np.array(mu.cpu()).flatten()
#         log_var = np.array(log_var.cpu()).flatten()
#         std = np.exp(log_var/2)
#         entropy_model = constriction.stream.model.QuantizedGaussian(z_int.min(),z_int.max())
        
#         coder = constriction.stream.stack.AnsCoder()
#         coder.encode_reverse(z_int, entropy_model, mu, std)
        
#         compressed = coder.get_compressed()

#         decoded = coder.decode(entropy_model, mu, std)
#         assert (z_int == decoded).all()
        
#         bit_sum += sum([(len(bin(word))-2) for word in compressed])
        
#         z_fp = dequantize(torch.tensor(decoded).to(device), z_scale)
#         Z[i,:] = z_fp
#     avg_bit_num = bit_sum / img_num
#     return avg_bit_num, Z
    # z = Z.flatten()
    # mu = MU.flatten()
    # log_var = LOG_VAR.flatten()
    # z_int, z_scale = quantize(z, qbit)
    # z_int = np.array(z_int.cpu()).flatten()
    # mu = np.array(mu.cpu()).flatten()
    # log_var = np.array(log_var.cpu()).flatten()
    # std = np.exp(log_var/2)
    # entropy_model = constriction.stream.model.QuantizedGaussian(z_int.min(),z_int.max())
    
    # coder = constriction.stream.stack.AnsCoder()
    # coder.encode_reverse(z_int, entropy_model, mu, std)
    
    # compressed = coder.get_compressed()

    # decoded = coder.decode(entropy_model, mu, std)
    # assert (z_int == decoded).all()
    
    # avg_bit_num = sum([(len(bin(word))-2) for word in compressed])/img_num
    
    # z_fp = dequantize(torch.tensor(decoded).to(device), z_scale)
    # Z = torch.reshape(z_fp, Z.shape)
    
    # return avg_bit_num, Z
        

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
    model = VAE(args.image_size, args.h_dim, args.latent_dim, args.z_dim).to(device)
    model.load_state_dict(torch.load('./out/vae/vae_org.pth', map_location=device))
    
    # load imags 'test_images.pth'
    img_num = 64
    img_size = (28, 28)
    imgs = torch.load('test_images.pth')  # size(64, 1, 28, 28)
    # save_image(imgs.view(64, 1, 28, 28), './out/figs/test_imgs' + '.png', nrow=8)

    with torch.no_grad():
        imgs = imgs.to(device)
        imgs = imgs.view(-1, args.image_size)
        print(imgs.shape)
        mu, log_var = model.encode(imgs)  # size(64, 2)
        mu_shape, log_var_shape = mu.shape, log_var.shape
        z = model.reparameterize(mu, log_var)
        x_prob = model.decode(z)

        qbits = [2,3,4,5,6,7,8,9,10]
        bit_vs_mse = np.zeros((len(qbits), 2))
        qbit_vs_bit = np.zeros((len(qbits), 2))
        for qbit in qbits:
            print('qbit:', qbit)
            avg_bit_num, mu_fp, log_var_fp = lossy_compress(mu, log_var, qbit)            
            z = model.reparameterize(mu_fp, log_var_fp)
            # avg_bit_num, z = lossy_compress(z, mu, log_var, qbit)
            imgs_fp = model.decode(z)
            imgs_fp = imgs_fp.view(-1, args.image_size)
            mse = cal_mse(imgs, imgs_fp).cpu()
            print('avg_bit_num:', avg_bit_num)
            print('mse:', mse)
            
            bit_vs_mse[qbits.index(qbit)] = [avg_bit_num, mse]
            imgs_fp = imgs_fp.view(img_num, 1, *img_size)
            imgs_fp = torch.round(imgs_fp)
            save_image(imgs_fp, './out/compression_lossy_vae/reconst_imgs_qbit' + str(qbit) + 'bit.png', nrow=8)

            # lossless compression
            avg_bit_num_lossless = lossless_compress(mu, log_var, qbit, imgs, imgs_fp, device)
            qbit_vs_bit[qbits.index(qbit)] = [qbit, avg_bit_num_lossless]
            
        plot_bit_vs_mse(bit_vs_mse)
        plot_qbit_vs_bit(qbit_vs_bit)        
        print(qbit_vs_bit)
        
        
        

    
    
    
    
    