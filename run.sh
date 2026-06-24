# 1. 进入 LLaMA-Factory 的 data 目录
cd /path/to/LLaMA-Factory/data  # 替换为你的实际路径

# 2. 下载两个极小的测试音频 (来自开源测试集)
wget -O test_audio_1.wav https://github.com/mozilla/TTS/raw/master/tests/data/ljspeech/wavs/LJ001-0001.wav
wget -O test_audio_2.wav https://github.com/mozilla/TTS/raw/master/tests/data/ljspeech/wavs/LJ001-0002.wav


[
  {
    "messages": [
      {
        "role": "user",
        "content": "<audio>\n请描述这段音频的内容。"
      },
      {
        "role": "assistant",
        "content": "这是一段测试音频，内容并不重要，主要是为了验证流程。"
      }
    ],
    "audio": "test_audio_1.wav"
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "<audio>\n这段声音是什么？"
      },
      {
        "role": "assistant",
        "content": "这是第二段测试音频。"
      }
    ],
    "audio": "test_audio_2.wav"
  }
]


  "test_audio_qwen": {
    "file_name": "test_audio_dataset.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "audio": "audio"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }




  #!/bin/bash

# 只用 1 张卡验证
export ASCEND_RT_VISIBLE_DEVICES=0

# 使用 torchrun 启动，nproc_per_node=1
torchrun --nproc_per_node=1 \
    src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2-Audio-7B-Instruct \
    --dataset test_audio_qwen \
    --template qwen2_audio \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ./output/verify_qwen2_audio \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --max_length 1024 \
    --bf16 true \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 1 \
    --overwrite_output_dir




