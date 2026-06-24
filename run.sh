# 1. 指定你要用的 4 张卡 (假设是 0,1,2,3 号卡)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# 2. 启动训练
torchrun --nproc_per_node=4 \
    swift/cli/train.py \
    --model_type qwen2-audio-7b-instruct \
    --model_id_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --dataset AI-ModelScope/Audio-FLAN-Dataset \
    --output_dir ./output/qwen2_audio_910b3 \
    \
    # --- 核心参数配置 ---
    --bf16 true \              # NPU 必须开 bf16
    --deepspeed zero2 \        # 显存优化，4张卡跑7B模型用 zero2 足够
    --per_device_train_batch_size 2 \ # 64GB显存，单卡跑2没问题
    --gradient_accumulation_steps 4 \ # 累积梯度，等效 Batch Size 为 2*4*4=32
    --learning_rate 1e-4 \
    --max_length 2048 \        # ️重要：限制最大长度防显存溢出，如有长音频可适当调小
    --lora_rank 16 \           # LoRA 秩，16兼顾效果和显存
    --num_train_epochs 3 \
    --logging_steps 5 \
    --save_steps 100 \
    --gradient_checkpointing true # 开启梯度检查点，进一步省显存
