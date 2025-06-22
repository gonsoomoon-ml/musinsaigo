#!/bin/bash

# CPU 훈련에 최적화된 SDXL LoRA 훈련 스크립트
echo "Starting SDXL LoRA training optimized for CPU..."

accelerate launch /home/ubuntu/musinsaigo/03_train/sdxl_lora/train.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --dataloader_num_workers 2 \
    --resolution 512 \
    --train_batch_size 1 \
    --learning_rate 1e-05 \
    --max_grad_norm 1 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --checkpointing_steps 5 \
    --seed 42 \
    --rank 4 \
    --num_train_epochs 1 \
    --random_flip \
    --snr_gamma 5.0 \
    --train_text_encoder \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
    --mixed_precision no \
    --train_data_dir /home/ubuntu/musinsaigo/test_data \
    --output_dir /home/ubuntu/musinsaigo/models \
    --report_to none

echo "Training completed!" 