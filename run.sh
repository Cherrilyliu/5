import os
import json
from pathlib import Path

def convert_aishell_to_llamafactory():
    """
    将 AISHELL-1 数据集转换为 LLaMA-Factory 格式
    """
    # AISHELL-1 数据路径
    aishell_dir = "/data/aishell1/data_aishell"
    wav_dir = os.path.join(aishell_dir, "wav")
    transcript_file = os.path.join(aishell_dir, "transcript", "aishell_transcript.txt")
    
    # 输出文件
    output_file = "/root/lgj/LLaMA-Factory/data/aishell_llama.json"
    
    # 读取转录文件
    print("正在读取转录文件...")
    transcripts = {}
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                audio_id = parts[0]
                text = ' '.join(parts[1:])
                transcripts[audio_id] = text
    
    print(f"共读取 {len(transcripts)} 条转录")
    
    # 查找所有 wav 文件
    print("正在查找音频文件...")
    wav_files = []
    for root, dirs, files in os.walk(wav_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    print(f"共找到 {len(wav_files)} 个音频文件")
    
    # 转换为 LLaMA-Factory 格式
    print("正在转换数据...")
    converted_data = []
    skipped = 0
    
    for wav_path in wav_files:
        # 从文件名提取 audio_id (例如: BAC009S0002W0121.wav -> BAC009S0002W0121)
        audio_id = Path(wav_path).stem
        
        # 查找对应的转录
        if audio_id not in transcripts:
            skipped += 1
            continue
        
        text = transcripts[audio_id]
        
        # 构造 LLaMA-Factory 格式
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<audio>\n请转录这段音频为文字。"
                },
                {
                    "role": "assistant",
                    "content": text
                }
            ],
            "audio": wav_path  # 使用绝对路径
        }
        
        converted_data.append(item)
    
    print(f"\n转换完成！")
    print(f"成功转换: {len(converted_data)} 条")
    print(f"跳过: {skipped} 条")
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"保存到: {output_file}")
    
    # 显示前 2 条示例
    print("\n数据示例（前 2 条）:")
    print(json.dumps(converted_data[:2], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    convert_aishell_to_llamafactory()
