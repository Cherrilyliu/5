import json
import os

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0" 

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import Audio
from jiwer import wer
import soundfile as sf
import librosa

# ================= 配置区 =================
BASE_MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"  # HF镜像上的基座模型ID或本地路径
LORA_WEIGHTS_PATH = "./output/qwen2_audio_aishell/checkpoint-xxx" # 你的LoRA权重绝对路径
TEST_JSON_PATH = "/root/lgj/LLaMA-Factory/data/aishell_llama.json" # 你的ShareGPT测试集路径
DEVICE = "npu:0" if torch.npu.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128
# ==========================================

def load_model_and_processor(base_path, lora_path):
    """兼容地加载基座模型 + LoRA权重"""
    print(f"正在加载基座模型: {base_path}")
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_path, 
        device_map=DEVICE,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32
    )
    
    # 如果LoRA路径存在，则合并权重
    if lora_path and os.path.exists(lora_path):
        print(f"正在合并LoRA权重: {lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload() # 关键：将LoRA权重合并到基座模型中
        
    model.eval()
    return model, processor

def extract_asr_data(json_path):
    """从ShareGPT格式中提取音频路径和真实文本"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_items = []
    for item in data:
        audio_path = item.get("audio")
        messages = item.get("messages", [])
        
        # 提取Assistant的真实转录文本作为Ground Truth
        ground_truth = ""
        for msg in messages:
            if msg["role"] == "assistant":
                ground_truth = msg["content"].strip()
                break
                
        if audio_path and ground_truth and os.path.exists(audio_path):
            test_items.append({"audio": audio_path, "gt": ground_truth})
            
    print(f"成功加载 {len(test_items)} 条有效测试数据")
    return test_items

def normalize_text(text):
    """WER计算前的文本标准化（去除标点、转小写等）"""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text) # 去除标点
    text = re.sub(r'\s+', ' ', text).strip() # 合并多余空格
    return text

def main():
    # 1. 加载模型
    model, processor = load_model_and_processor(BASE_MODEL_PATH, LORA_WEIGHTS_PATH)
    
    # 2. 加载测试数据
    test_data = extract_asr_data(TEST_JSON_PATH)
    
    predictions = []
    references = []
    
    # 3. 逐条推理
    for i, item in enumerate(test_data):
        try:
            # 使用soundfile/librosa安全加载音频，避免processor直接读路径可能出现的兼容性问题
            waveform, sr = librosa.load(item["audio"], sr=16000) 
            
            inputs = processor(
                text="<audio>", # Qwen2-Audio的音频触发token
                audios=[waveform],
                sampling_rate=16000,
                return_tensors="pt"
            ).to(DEVICE)
            
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 清洗预测文本（去掉可能生成的前缀提示词）
            if "assistant" in pred_text.lower():
                pred_text = pred_text.split("assistant")[-1].strip()
                
            predictions.append(normalize_text(pred_text))
            references.append(normalize_text(item["gt"]))
            
            if (i + 1) % 50 == 0:
                print(f"进度: {i+1}/{len(test_data)} | 当前WER: {wer(references, predictions):.4f}")
                
        except Exception as e:
            print(f"处理第 {i} 条数据时出错: {e}")
            continue
            
    # 4. 计算最终WER
    final_wer = wer(references, predictions)
    print("\n" + "="*40)
    print(f" 评估完成！")
    print(f"📊 总测试样本数: {len(predictions)}")
    print(f" Word Error Rate (WER): {final_wer:.4f} ({final_wer*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    main()
