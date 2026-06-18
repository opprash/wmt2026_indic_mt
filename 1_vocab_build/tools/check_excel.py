import pandas as pd
import json
import os
import sys

def check_excel_structure(file_path):
    """检查Excel文件结构"""
    print(f"检查文件: {file_path}")

    try:
        # 读取Excel文件
        df = pd.read_excel(file_path, nrows=5)

        print(f"文件路径: {file_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {', '.join(df.columns.tolist())}")

        # 尝试识别英文列和目标语言列
        eng_columns = []
        target_columns = []

        for col in df.columns:
            col_str = str(col).lower()

            # 检查是否是英文列
            if 'english' in col_str or 'eng' in col_str or 'en' in col_str or 'source' in col_str:
                eng_columns.append(col)
            # 检查是否是目标语言列
            elif 'karbi' in col_str or 'bodo' in col_str or 'kokborok' in col_str or 'nagamese' in col_str or 'tagin' in col_str:
                target_columns.append(col)
            # 如果只有两列，则第一列是英文，第二列是目标语言
            elif len(df.columns) == 2:
                if df.columns.get_loc(col) == 0:
                    eng_columns.append(col)
                else:
                    target_columns.append(col)

        print(f"识别的英文列: {eng_columns}")
        print(f"识别的目标语言列: {target_columns}")

        print("\n前5行数据预览:")
        print(df.head())

        print("\n" + "="*80 + "\n")

        return eng_columns, target_columns

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return [], []

# 检查所有Excel文件
excel_files = [
    'C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/Category II/English - Karbi Training Data 2026.xlsx',
    'C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/Category II/English-Bodo Training Data 2026.xlsx',
    'C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/Category II/English-Kokborok Training Data 2026.xlsx',
    'C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/Category II/English-Nagamese  Training Data 2026.xlsx',
    'C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/Category II/English-Tagin Training Data 2026.xlsx'
]

all_results = {}
for file in excel_files:
    if os.path.exists(file):
        eng_cols, target_cols = check_excel_structure(file)
        file_name = os.path.basename(file)
        all_results[file_name] = {
            'eng_columns': eng_cols,
            'target_columns': target_cols
        }
    else:
        print(f"文件不存在: {file}")

# 保存检查结果
with open('C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/code/excel_structure.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n检查结果已保存到: C:/Users/Administrator/Desktop/智能2026/个人/wmt2026/code/excel_structure.json")