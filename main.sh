CUDA_VISIBLE_DEVICES=3
#!/bin/bash
IMAGE_SIZE=784
H_DIM=400
LATENT_DIM=20
Z_DIM=20
DATA_DIR="../data"
BATCH_SIZE=128
EPOCHS=2000
GPU=2,3
SEED=66
LOG_INTERVAL=100
LR=1e-3

python main.py \
  --image-size $IMAGE_SIZE \
  --h-dim $H_DIM \
  --latent-dim $LATENT_DIM \
  --z-dim $Z_DIM \
  --data-dir $DATA_DIR \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --gpu $GPU \
  --seed $SEED \
  --log-interval $LOG_INTERVAL \
  --lr $LR 