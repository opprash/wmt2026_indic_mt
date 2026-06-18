#!/usr/bin/env python3
"""
工具脚本用于读取Excel文件并显示数据结构
"""

import os
import sys
import json

def read_excel_info():
    """读取Excel文件信息的简单方法"""
    excel_dir = "C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/Category II"
    files = [
        "English - Karbi Training Data 2026.xlsx",
        "English-Nagamese Training Data 2026.xlsx",
        "English-Bodo Training Data 2026.xlsx",
        "English-Kokborok Training Data 2026.xlsx",
        "English-Tagin Training Data 2026.xlsx"
    ]

    print("Excel文件列表:")
    for file in files:
        full_path = os.path.join(excel_dir, file.replace(" ", " "))
        if os.path.exists(full_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (未找到)")
            # 尝试查找正确的文件名
            actual_files = [f for f in os.listdir(excel_dir) if file.lower() in f.lower()]
            if actual_files:
                print(f"    可能匹配: {actual_files}")

    print("\n数据格式说明:")
    print("根据截图，数据可能包含以下列:")
    print("1. English text - 英文文本")
    print("2. Target language text - 目标语言文本")
    print("3. 可能的其他列: Source, Target, 翻译质量等")

    return files

if __name__ == "__main__":
    read_excel_info()