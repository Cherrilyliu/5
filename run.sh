import json
import os
from pathlib import Path
from datasets import load_dataset

def convert_audio_flan():
    """
    将 Audio-FLAN 数据集转换为 LLaMA-Factory 的 ShareGPT 格式
    """
    
    # 数据集路径
    dataset_path = "./data/audio_flan_raw"
    output_file = "./data/audio_flan_llama.json"
    
    print("正在加载 Audio-FLAN 数据集...")
    
    # 加载数据集（根据实际下载的文件格式调整）
    try:
        # 尝试加载 parquet 格式
        dataset = load_dataset("parquet", data_dir=dataset_path, split="train")
    except:
        try:
            # 尝试加载 jsonl 格式
            dataset = load_dataset("json", data_files=f"{dataset_path}/train.jsonl", split="train")
        except:
            # 尝试加载 json 格式
            dataset = load_dataset("json", data_files=f"{dataset_path}/train.json", split="train")
    
    print(f"数据集加载完成，共 {len(dataset)} 条数据")
    
    converted_data = []
    skipped_count = 0
    
    print("开始转换数据...")
    
    for i, item in enumerate(dataset):
        try:
            # 获取音频路径
            # Audio-FLAN 的音频字段可能是 'audio'，结构可能是 dict 或 string
            audio_info = item.get("audio")
            
            if isinstance(audio_info, dict):
                # 如果是字典，提取 path 或 bytes
                audio_path = audio_info.get("path")
                if not audio_path:
                    # 如果没有 path，可能需要保存 bytes 到文件
                    skipped_count += 1
                    continue
            elif isinstance(audio_info, str):
                audio_path = audio_info
            else:
                skipped_count += 1
                continue
            
            # 转换为绝对路径
            if not os.path.isabs(audio_path):
                audio_path = os.path.abspath(os.path.join(dataset_path, audio_path))
            
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"警告：音频文件不存在: {audio_path}")
                skipped_count += 1
                continue
            
            # 获取指令和回复
            # Audio-FLAN 可能的字段名
            instruction = (item.get("instruction") or 
                          item.get("prompt") or 
                          item.get("question") or 
                          item.get("input") or 
                          "")
            
            response = (item.get("response") or 
                       item.get("output") or 
                       item.get("answer") or 
                       "")
            
            if not instruction or not response:
                skipped_count += 1
                continue
            
            # 构造 LLaMA-Factory 格式
            converted_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<audio>\n{instruction.strip()}"
                    },
                    {
                        "role": "assistant",
                        "content": response.strip()
                    }
                ],
                "audio": audio_path
            }
            
            converted_data.append(converted_item)
            
            # 每 1000 条打印进度
            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1}/{len(dataset)} 条数据...")
                
        except Exception as e:
            print(f"处理第 {i} 条数据时出错: {e}")
            skipped_count += 1
            continue
    
    # 保存转换后的数据
    print(f"\n转换完成！")
    print(f"成功转换: {len(converted_data)} 条")
    print(f"跳过: {skipped_count} 条")
    print(f"保存到: {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("保存完成！")
    
    # 显示前 2 条数据示例
    print("\n数据示例（前 2 条）:")
    print(json.dumps(converted_data[:2], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    convert_audio_flan()
