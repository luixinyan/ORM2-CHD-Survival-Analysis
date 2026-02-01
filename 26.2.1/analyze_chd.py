#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
冠心病患者ORM2与结局事件关系分析
包括MACE和全因死亡两类结局
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 读取数据
file_path = r'd:\onedrive\博士\博三\3-ORM2-第二次ELISA后开始最后分析\26.1.18\26.2.1\13.单独列出的冠心病.xlsx'
df = pd.read_excel(file_path)

print("=" * 80)
print("数据基本信息")
print("=" * 80)
print(f'数据集形状: {df.shape[0]} 行 × {df.shape[1]} 列\n')

print("列名列表:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "=" * 80)
print("前10行数据预览")
print("=" * 80)
print(df.head(10))

print("\n" + "=" * 80)
print("数据类型")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("缺失值统计")
print("=" * 80)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '缺失数': missing,
    '缺失率(%)': missing_pct
})
print(missing_df[missing_df['缺失数'] > 0])

print("\n" + "=" * 80)
print("数值型变量描述性统计")
print("=" * 80)
print(df.describe())

# 寻找可能的结局变量
print("\n" + "=" * 80)
print("查找可能的结局变量（MACE、全因死亡相关）")
print("=" * 80)
outcome_keywords = ['MACE', 'mace', 'death', 'Death', '死亡', 'mortality', 'Mortality',
                    'outcome', 'Outcome', 'event', 'Event', '事件', '结局']
outcome_cols = [col for col in df.columns if any(keyword in str(col) for keyword in outcome_keywords)]
if outcome_cols:
    print(f"找到可能的结局变量: {outcome_cols}")
    for col in outcome_cols:
        print(f"\n{col} 的值分布:")
        print(df[col].value_counts())
else:
    print("未找到明显的结局变量，请检查列名")

# 查找ORM2变量
print("\n" + "=" * 80)
print("查找ORM2变量")
print("=" * 80)
orm2_cols = [col for col in df.columns if 'ORM2' in str(col) or 'orm2' in str(col)]
if orm2_cols:
    print(f"找到ORM2相关变量: {orm2_cols}")
    for col in orm2_cols:
        print(f"\n{col} 的描述性统计:")
        print(df[col].describe())
else:
    print("未找到ORM2变量，请检查列名")
