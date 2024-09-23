CUDA_VISIBLE_DEVICES=3
#!/bin/bash
IMAGE_SIZE=784
H_DIM=400
Z_DIM=20
GPU=2,3

python eval.py \
  --image-size $IMAGE_SIZE \
  --h-dim $H_DIM \
  --z-dim $Z_DIM \
  --gpu $GPU 