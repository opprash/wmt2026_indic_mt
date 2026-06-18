#!/usr/bin/env python3
"""
分析Excel数据结构的简单脚本
"""
import os

def check_excel_files():
    """检查Excel文件的存在和结构"""
    base_dir = "C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/Category II"

    # Excel文件列表
    excel_files = [
        "English - Karbi Training Data 2026.xlsx",
        "English-Bodo Training Data 2026.xlsx",
        "English-Kokborok Training Data 2026.xlsx",
        "English-Nagamese Training Data 2026.xlsx",
        "English-Tagin Training Data 2026.xlsx"
    ]

    print("=== Excel文件检查 ===")
    all_files = []
    for file in excel_files:
        path = os.path.join(base_dir, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"[OK] {file}: {size:.2f} MB")
            all_files.append(file)
        else:
            print(f"[Missing] {file}")

    print(f"\n总共找到 {len(all_files)} 个Excel文件")

    print("\n=== 数据格式假设 ===")
    print("基于您的描述和截图，预计每个Excel文件包含:")
    print("1. Source (English) - 英文句子")
    print("2. Target - 目标语言句子")
    print("3. 可能的其他列: context, domain, quality_score等")

    return all_files

def print_language_info():
    """显示语言信息"""
    print("\n=== 目标语言信息 ===")
    languages = {
        "Karbi": "印度东北部阿萨姆邦的藏缅语系语言 (卡比语)",
        "Bodo": "印度阿萨姆邦的主要语言之一 (博多语)",
        "Kokborok": "特里普拉邦的主要语言 (科克博罗克语)",
        "Nagamese": "那加兰邦的克里奥尔语 (那加语)",
        "Tagin": "阿鲁纳恰尔邦的藏缅语系语言 (达金语)"
    }

    for lang, desc in languages.items():
        print(f"{lang}: {desc}")

    print("\n所有语言都属于英语-小语种低资源翻译任务")

if __name__ == "__main__":
    files = check_excel_files()
    print_language_info()