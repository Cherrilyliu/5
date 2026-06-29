#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 qwen3-32B 模型清洗 instruction 字段
"""

import json
import re
from typing import Dict, List
import time

def call_qwen_api(instruction: str, api_base: str = "http://localhost:11434/api") -> str:
    """
    调用本地 qwen3-32B 模型

    Args:
        instruction: 需要清洗的instruction
        api_base: API地址

    Returns:
        清洗后的instruction
    """
    url = f"{api_base}/chat/completions"

    prompt = f"""你是一位专业的指令清洗专家。你的任务是将输入的instruction清洗成规范、清晰、自然的表述。

请只输出清洗后的instruction，不要有任何解释、前缀或后缀。

清洗规则：
1. 修复语法错误和杂糅结构
2. 去除重复修饰（如"怎样...的..."、"什么是...的..."等）
3. 使表述更加简洁明了
4. 保持原意不变
5. 如果instruction已经很规范，则原样返回

输入instruction：{instruction}
"""

    payload = {
        "model": "qwen3-32b",
        "messages": [
            {"role": "system", "content": "你是一个专业的指令清洗助手，只输出清洗后的instruction。"},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.1,  # 降低随机性，确保一致性
        "max_tokens": 1000
    }

    try:
        import requests
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  调用模型失败: {e}")
        return instruction


def clean_dataset_with_qwen(
    input_file: str,
    output_file: str,
    api_base: str = "http://localhost:11434/api",
    batch_size: int = 10,
    delay: float = 1.0
):
    """
    使用 qwen3-32B 模型清洗整个数据集

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        api_base: Ollama API地址
        batch_size: 批量处理大小（0表示逐个处理）
        delay: 每次请求之间的延迟（秒）
    """
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"数据集总样本数: {total_samples}")

    # 跟踪清洗进度
    cleaned_instructions = []
    unchanged_count = 0
    change_count = 0

    # 统计问题类型
    issue_types = {
        "语法杂糅_为什么要是什么": 0,
        "重复修饰_怎样...的...": 0,
        "重复修饰_什么是...的...": 0,
        "重复修饰_请简要概括...具有什么": 0,
        "格式不规范": 0,
        "其他问题": 0
    }

    print("\n开始清洗...")
    print("="*60)

    for i, item in enumerate(data):
        if 'instruction' in item and item['instruction']:
            original = item['instruction']
            cleaned = call_qwen_api(original, api_base)

            # 统计问题类型
            if original != cleaned:
                change_count += 1
                issue_types["其他问题"] += 1

                # 判断问题类型
                if "为什么要" in original and "是什么" in original:
                    issue_types["语法杂糅_为什么要是什么"] += 1
                elif "怎样" in original and "的" in original:
                    issue_types["重复修饰_怎样...的..."] += 1
                elif "什么是" in original and "的" in original:
                    issue_types["重复修饰_什么是...的..."] += 1
                elif "请简要概括" in original and "具有什么" in original:
                    issue_types["重复修饰_请简要概括...具有什么"] += 1
                elif not re.match(r'^[请说明解释阐述介绍概括]\s+.*$', original):
                    issue_types["格式不规范"] += 1
            else:
                unchanged_count += 1

            # 更新instruction
            item['instruction'] = cleaned
            cleaned_instructions.append((i, original, cleaned))

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{total_samples} ({(i+1)/total_samples*100:.1f}%)")

            # 请求延迟，避免过载
            time.sleep(delay)

    # 保存清洗后的数据
    print("\n" + "="*60)
    print("正在保存清洗后的数据: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print("\n" + "="*60)
    print("清洗统计:")
    print("="*60)
    print(f"总样本数: {total_samples}")
    print(f"清洗后修改的样本数: {change_count}")
    print(f"清洗后保持不变的样本数: {unchanged_count}")
    print(f"\n问题类型分布:")
    for issue_type, count in issue_types.items():
        if count > 0:
            percentage = (count / change_count) * 100 if change_count > 0 else 0
            print(f"  {issue_type}: {count} ({percentage:.1f}%)")
    print("="*60)

    # 打印修改示例
    print(f"\n修改示例（前{min(10, change_count)}个）:")
    print("="*60)
    for idx, (i, original, cleaned) in enumerate(cleaned_instructions[:10]):
        print(f"\n[{idx+1}] 原文: {original}")
        print(f"      修改: {cleaned}")
    print("="*60)

    print("\n✓ 清洗完成！")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='使用 qwen3-32B 模型清洗数据集')
    parser.add_argument('--input', type=str, default=r'c:\Users\23833\Desktop\final_dataset(1).json',
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default=r'c:\Users\23833\Desktop\final_dataset_qwen_cleaned.json',
                        help='输出文件路径')
    parser.add_argument('--api', type=str, default='http://localhost:11434/api',
                        help='Ollama API地址')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='每次请求之间的延迟（秒）')

    args = parser.parse_args()

    clean_dataset_with_qwen(
        input_file=args.input,
        output_file=args.output,
        api_base=args.api,
        delay=args.delay
    )
