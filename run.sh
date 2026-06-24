#!/bin/bash

# 1. 指定你要用的 4 张卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# 2. 启动训练 (直接使用 swift sft 命令，不需要 torchrun)
swift sft \
    --model_type qwen2-audio-7b-instruct \
    --model_id_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --dataset AI-ModelScope/Audio-FLAN-Dataset \
    --output_dir ./output/qwen2_audio_910b3 \
    --bf16 true \
    --deepspeed zero2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --lora_rank 16 \
    --num_train_epochs 3 \
    --logging_steps 5 \
    --save_steps 100 \
    --gradient_checkpointing true
