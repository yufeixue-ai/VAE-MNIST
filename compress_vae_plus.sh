CUDA_VISIBLE_DEVICES=3
#!/bin/bash
IMAGE_SIZE=784
H_DIM=400
LATENT_DIM=20
Z_DIM=2
GPU=2,3

python compress_vae_plus.py \
  --image-size $IMAGE_SIZE \
  --h-dim $H_DIM \
  --latent-dim $LATENT_DIM \
  --z-dim $Z_DIM \
  --gpu $GPU 