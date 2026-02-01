#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ORM2四分位数与MACE - 改进的多因素Cox回归分析
使用逐步添加协变量的方法来解决收敛问题
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 读取数据
file_path = r'd:\onedrive\博士\博三\3-ORM2-第二次ELISA后开始最后分析\26.1.18\26.2.1\13.单独列出的冠心病.xlsx'
df = pd.read_excel(file_path)

print("=" * 100)
print("ORM2四分位数与MACE - 改进的多因素Cox回归分析")
print("=" * 100)

# ORM2四分位数分组
df['ORM2_quartile'] = pd.qcut(df['ORM2'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 准备数据并检查缺失情况
print("\n【数据准备】")
print("\n各变量缺失情况:")
key_vars = ['ORM2_quartile', 'MACE_随访天数', 'MACE', '年龄', '性别', 'BMI',
            '糖尿病', '高血压', '吸烟史', '饮酒史']
for var in key_vars:
    if var in df.columns:
        missing = df[var].isnull().sum()
        print(f"{var}: {missing} ({missing/len(df)*100:.1f}%)")

# 创建分析数据集
df_cox = df[['ORM2_quartile', 'MACE_随访天数', 'MACE', '年龄', '性别', 'BMI']].copy()
df_cox = df_cox.dropna()
print(f"\n完整病例分析样本量: {len(df_cox)} 例")
print(f"MACE事件: {df_cox['MACE'].sum()} 例")

# 创建哑变量
df_cox = pd.get_dummies(df_cox, columns=['ORM2_quartile'], drop_first=True, dtype=int)
df_cox.columns = ['duration', 'event', '年龄', '性别', 'BMI', 'Q2', 'Q3', 'Q4']

# 检查协变量分布
print("\n协变量描述性统计:")
print(df_cox[['年龄', '性别', 'BMI']].describe())

# 检查异常值
print("\n检查异常值:")
for var in ['年龄', 'BMI']:
    q1 = df_cox[var].quantile(0.25)
    q3 = df_cox[var].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    outliers = df_cox[(df_cox[var] < lower) | (df_cox[var] > upper)]
    print(f"{var}: {len(outliers)} 个极端异常值 (< {lower:.1f} 或 > {upper:.1f})")

print("\n" + "=" * 100)
print("逐步多因素Cox回归分析")
print("=" * 100)

# 模型1: 粗模型（仅ORM2四分位数）
print("\n【模型1: 粗模型】")
print("变量: ORM2四分位数")
cph_m1 = CoxPHFitter()
cph_m1.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4']],
           duration_col='duration', event_col='event')

print("\nHR (95% CI):")
print(f"Q1: 1.00 (reference)")
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m1.summary.loc[q, 'exp(coef)']
    ci_l = cph_m1.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m1.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m1.summary.loc[q, 'p']
    print(f"{q}: {hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), P={p:.4f}")

# 模型2: 校正年龄
print("\n【模型2: 校正年龄】")
print("变量: ORM2四分位数 + 年龄")
try:
    df_cox_m2 = df_cox.copy()
    scaler = StandardScaler()
    df_cox_m2['年龄_标准化'] = scaler.fit_transform(df_cox[['年龄']])

    cph_m2 = CoxPHFitter()
    cph_m2.fit(df_cox_m2[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄_标准化']],
               duration_col='duration', event_col='event')

    print("\nHR (95% CI):")
    print(f"Q1: 1.00 (reference)")
    for q in ['Q2', 'Q3', 'Q4']:
        hr = cph_m2.summary.loc[q, 'exp(coef)']
        ci_l = cph_m2.summary.loc[q, 'exp(coef) lower 95%']
        ci_u = cph_m2.summary.loc[q, 'exp(coef) upper 95%']
        p = cph_m2.summary.loc[q, 'p']
        print(f"{q}: {hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), P={p:.4f}")

    hr_age = cph_m2.summary.loc['年龄_标准化', 'exp(coef)']
    p_age = cph_m2.summary.loc['年龄_标准化', 'p']
    print(f"\n年龄（每增加1SD）: HR={hr_age:.2f}, P={p_age:.4f}")

    model2_success = True
except Exception as e:
    print(f"模型2失败: {e}")
    model2_success = False

# 模型3: 校正年龄 + 性别
print("\n【模型3: 校正年龄 + 性别】")
print("变量: ORM2四分位数 + 年龄 + 性别")
if model2_success:
    try:
        cph_m3 = CoxPHFitter()
        cph_m3.fit(df_cox_m2[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄_标准化', '性别']],
                   duration_col='duration', event_col='event')

        print("\nHR (95% CI):")
        print(f"Q1: 1.00 (reference)")
        for q in ['Q2', 'Q3', 'Q4']:
            hr = cph_m3.summary.loc[q, 'exp(coef)']
            ci_l = cph_m3.summary.loc[q, 'exp(coef) lower 95%']
            ci_u = cph_m3.summary.loc[q, 'exp(coef) upper 95%']
            p = cph_m3.summary.loc[q, 'p']
            print(f"{q}: {hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), P={p:.4f}")

        hr_sex = cph_m3.summary.loc['性别', 'exp(coef)']
        p_sex = cph_m3.summary.loc['性别', 'p']
        print(f"\n性别（男vs女）: HR={hr_sex:.2f}, P={p_sex:.4f}")

        model3_success = True
    except Exception as e:
        print(f"模型3失败: {e}")
        model3_success = False
else:
    print("由于模型2失败，跳过模型3")
    model3_success = False

# 模型4: 完全校正模型（年龄 + 性别 + BMI）
print("\n【模型4: 完全校正模型】")
print("变量: ORM2四分位数 + 年龄 + 性别 + BMI")
if model3_success:
    try:
        df_cox_m4 = df_cox.copy()
        df_cox_m4[['年龄_标准化', 'BMI_标准化']] = scaler.fit_transform(df_cox[['年龄', 'BMI']])

        cph_m4 = CoxPHFitter(penalizer=0.01)  # 添加L2惩罚以改善收敛
        cph_m4.fit(df_cox_m4[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄_标准化', '性别', 'BMI_标准化']],
                   duration_col='duration', event_col='event')

        print("\nHR (95% CI):")
        print(f"Q1: 1.00 (reference)")
        for q in ['Q2', 'Q3', 'Q4']:
            hr = cph_m4.summary.loc[q, 'exp(coef)']
            ci_l = cph_m4.summary.loc[q, 'exp(coef) lower 95%']
            ci_u = cph_m4.summary.loc[q, 'exp(coef) upper 95%']
            p = cph_m4.summary.loc[q, 'p']
            print(f"{q}: {hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), P={p:.4f}")

        print("\n协变量:")
        hr_age = cph_m4.summary.loc['年龄_标准化', 'exp(coef)']
        p_age = cph_m4.summary.loc['年龄_标准化', 'p']
        hr_sex = cph_m4.summary.loc['性别', 'exp(coef)']
        p_sex = cph_m4.summary.loc['性别', 'p']
        hr_bmi = cph_m4.summary.loc['BMI_标准化', 'exp(coef)']
        p_bmi = cph_m4.summary.loc['BMI_标准化', 'p']

        print(f"年龄（每增加1SD）: HR={hr_age:.2f}, P={p_age:.4f}")
        print(f"性别（男vs女）: HR={hr_sex:.2f}, P={p_sex:.4f}")
        print(f"BMI（每增加1SD）: HR={hr_bmi:.2f}, P={p_bmi:.4f}")

        model4_success = True
    except Exception as e:
        print(f"模型4失败: {e}")
        print("\n尝试不使用惩罚项...")
        try:
            cph_m4_alt = CoxPHFitter()
            cph_m4_alt.fit(df_cox_m4[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄_标准化', '性别', 'BMI_标准化']],
                           duration_col='duration', event_col='event')

            print("\nHR (95% CI):")
            print(f"Q1: 1.00 (reference)")
            for q in ['Q2', 'Q3', 'Q4']:
                hr = cph_m4_alt.summary.loc[q, 'exp(coef)']
                ci_l = cph_m4_alt.summary.loc[q, 'exp(coef) lower 95%']
                ci_u = cph_m4_alt.summary.loc[q, 'exp(coef) upper 95%']
                p = cph_m4_alt.summary.loc[q, 'p']
                print(f"{q}: {hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), P={p:.4f}")

            cph_m4 = cph_m4_alt
            model4_success = True
        except Exception as e2:
            print(f"替代方法也失败: {e2}")
            model4_success = False
else:
    print("由于模型3失败，跳过模型4")
    model4_success = False

# 生成汇总表
print("\n" + "=" * 100)
print("各模型结果汇总")
print("=" * 100)

summary_data = {
    'ORM2分组': ['Q1', 'Q2', 'Q3', 'Q4'],
    '模型1_粗HR': ['1.00 (ref)'],
    '模型2_校正年龄': ['1.00 (ref)'] if model2_success else ['1.00 (ref)', '-', '-', '-'],
    '模型3_校正年龄性别': ['1.00 (ref)'] if model3_success else ['1.00 (ref)', '-', '-', '-'],
    '模型4_完全校正': ['1.00 (ref)'] if model4_success else ['1.00 (ref)', '-', '-', '-']
}

# 填充模型1
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m1.summary.loc[q, 'exp(coef)']
    ci_l = cph_m1.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m1.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m1.summary.loc[q, 'p']
    p_str = f"P={p:.3f}" if p >= 0.001 else "P<0.001"
    summary_data['模型1_粗HR'].append(f"{hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), {p_str}")

# 填充模型2
if model2_success:
    for q in ['Q2', 'Q3', 'Q4']:
        hr = cph_m2.summary.loc[q, 'exp(coef)']
        ci_l = cph_m2.summary.loc[q, 'exp(coef) lower 95%']
        ci_u = cph_m2.summary.loc[q, 'exp(coef) upper 95%']
        p = cph_m2.summary.loc[q, 'p']
        p_str = f"P={p:.3f}" if p >= 0.001 else "P<0.001"
        summary_data['模型2_校正年龄'].append(f"{hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), {p_str}")

# 填充模型3
if model3_success:
    for q in ['Q2', 'Q3', 'Q4']:
        hr = cph_m3.summary.loc[q, 'exp(coef)']
        ci_l = cph_m3.summary.loc[q, 'exp(coef) lower 95%']
        ci_u = cph_m3.summary.loc[q, 'exp(coef) upper 95%']
        p = cph_m3.summary.loc[q, 'p']
        p_str = f"P={p:.3f}" if p >= 0.001 else "P<0.001"
        summary_data['模型3_校正年龄性别'].append(f"{hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), {p_str}")

# 填充模型4
if model4_success:
    for q in ['Q2', 'Q3', 'Q4']:
        hr = cph_m4.summary.loc[q, 'exp(coef)']
        ci_l = cph_m4.summary.loc[q, 'exp(coef) lower 95%']
        ci_u = cph_m4.summary.loc[q, 'exp(coef) upper 95%']
        p = cph_m4.summary.loc[q, 'p']
        p_str = f"P={p:.3f}" if p >= 0.001 else "P<0.001"
        summary_data['模型4_完全校正'].append(f"{hr:.2f} ({ci_l:.2f}-{ci_u:.2f}), {p_str}")

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# 保存到Excel
output_file = 'ORM2四分位数_多模型Cox回归结果.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='汇总结果', index=False)
    cph_m1.summary.to_excel(writer, sheet_name='模型1_粗模型')
    if model2_success:
        cph_m2.summary.to_excel(writer, sheet_name='模型2_校正年龄')
    if model3_success:
        cph_m3.summary.to_excel(writer, sheet_name='模型3_校正年龄性别')
    if model4_success:
        cph_m4.summary.to_excel(writer, sheet_name='模型4_完全校正')

print(f"\n结果已保存至: {output_file}")

print("\n" + "=" * 100)
print("分析完成！")
print("=" * 100)
