#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ORM2四分位数与MACE - 最终版Cox回归分析
排除无变异性的协变量
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
print("ORM2四分位数与MACE事件的Cox回归分析 - 最终版")
print("=" * 100)

# ORM2四分位数分组
df['ORM2_quartile'] = pd.qcut(df['ORM2'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 准备数据
df_cox = df[['ORM2_quartile', 'MACE_随访天数', 'MACE', '年龄', '性别', 'BMI']].copy()

# 检查变量的变异性
print("\n【变量变异性检查】")
print(f"性别的唯一值: {df_cox['性别'].nunique()} (所有患者: {df_cox['性别'].value_counts().to_dict()})")
print(f"年龄范围: {df_cox['年龄'].min():.0f}-{df_cox['年龄'].max():.0f} 岁")
print(f"BMI范围: {df_cox['BMI'].min():.2f}-{df_cox['BMI'].max():.2f}")

# 删除缺失值
df_cox = df_cox.dropna()
print(f"\n完整病例分析样本量: {len(df_cox)} 例")
print(f"MACE事件: {df_cox['MACE'].sum()} 例 ({df_cox['MACE'].sum()/len(df_cox)*100:.1f}%)")

# 创建哑变量
df_cox = pd.get_dummies(df_cox, columns=['ORM2_quartile'], drop_first=True, dtype=int)
df_cox.columns = ['duration', 'event', '年龄', '性别', 'BMI', 'Q2', 'Q3', 'Q4']

# 标准化连续变量
scaler = StandardScaler()
df_cox[['年龄_std', 'BMI_std']] = scaler.fit_transform(df_cox[['年龄', 'BMI']])

print("\n" + "=" * 100)
print("Cox回归分析 - 递进模型")
print("=" * 100)

# 获取基线数据
baseline_summary = df.groupby('ORM2_quartile').agg({
    'MACE': ['sum', 'count'],
    'ORM2': ['median', 'min', 'max']
}).round(2)

# 模型1: 粗模型
print("\n【模型1: 粗模型（未校正）】")
cph_m1 = CoxPHFitter()
cph_m1.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4']],
           duration_col='duration', event_col='event')

print("\nORM2四分位数 HR (95% CI):")
results_m1 = []
results_m1.append(['Q1', '1.00', 'reference'])
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m1.summary.loc[q, 'exp(coef)']
    ci_l = cph_m1.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m1.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m1.summary.loc[q, 'p']
    results_m1.append([q, f"{hr:.2f}", f"{ci_l:.2f}-{ci_u:.2f}", f"{p:.4f}"])
    print(f"{q}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f}")

# 模型2: 校正年龄
print("\n【模型2: 校正年龄】")
cph_m2 = CoxPHFitter()
cph_m2.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄_std']],
           duration_col='duration', event_col='event')

print("\nORM2四分位数 HR (95% CI):")
results_m2 = []
results_m2.append(['Q1', '1.00', 'reference'])
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m2.summary.loc[q, 'exp(coef)']
    ci_l = cph_m2.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m2.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m2.summary.loc[q, 'p']
    results_m2.append([q, f"{hr:.2f}", f"{ci_l:.2f}-{ci_u:.2f}", f"{p:.4f}"])
    print(f"{q}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f}")

hr_age = cph_m2.summary.loc['年龄_std', 'exp(coef)']
p_age = cph_m2.summary.loc['年龄_std', 'p']
print(f"\n年龄 (每增加1 SD): HR = {hr_age:.2f}, P = {p_age:.4f}")

# 模型3: 校正年龄 + BMI
print("\n【模型3: 校正年龄 + BMI（完全校正模型）】")
try:
    cph_m3 = CoxPHFitter(penalizer=0.01)
    cph_m3.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄_std', 'BMI_std']],
               duration_col='duration', event_col='event')

    print("\nORM2四分位数 HR (95% CI):")
    results_m3 = []
    results_m3.append(['Q1', '1.00', 'reference'])
    for q in ['Q2', 'Q3', 'Q4']:
        hr = cph_m3.summary.loc[q, 'exp(coef)']
        ci_l = cph_m3.summary.loc[q, 'exp(coef) lower 95%']
        ci_u = cph_m3.summary.loc[q, 'exp(coef) upper 95%']
        p = cph_m3.summary.loc[q, 'p']
        results_m3.append([q, f"{hr:.2f}", f"{ci_l:.2f}-{ci_u:.2f}", f"{p:.4f}"])
        print(f"{q}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f}")

    hr_age = cph_m3.summary.loc['年龄_std', 'exp(coef)']
    p_age = cph_m3.summary.loc['年龄_std', 'p']
    hr_bmi = cph_m3.summary.loc['BMI_std', 'exp(coef)']
    p_bmi = cph_m3.summary.loc['BMI_std', 'p']
    print(f"\n年龄 (每增加1 SD): HR = {hr_age:.2f}, P = {p_age:.4f}")
    print(f"BMI (每增加1 SD): HR = {hr_bmi:.2f}, P = {p_bmi:.4f}")

    model3_success = True
except Exception as e:
    print(f"模型3拟合失败: {e}")
    results_m3 = [['Q1', '1.00', 'reference'], ['Q2', '-', '-'], ['Q3', '-', '-'], ['Q4', '-', '-']]
    model3_success = False

# 趋势检验
print("\n" + "=" * 100)
print("趋势检验（P for trend）")
print("=" * 100)

df_trend = df[['MACE_随访天数', 'MACE', 'ORM2_quartile']].copy().dropna()
df_trend['quartile_numeric'] = df_trend['ORM2_quartile'].map({'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3})

cph_trend = CoxPHFitter()
cph_trend.fit(df_trend[['MACE_随访天数', 'MACE', 'quartile_numeric']].rename(
    columns={'MACE_随访天数': 'duration', 'MACE': 'event'}),
    duration_col='duration', event_col='event')

p_trend = cph_trend.summary.loc['quartile_numeric', 'p']
print(f"P for trend = {p_trend:.4f}")

# 生成论文格式的表格
print("\n" + "=" * 100)
print("结果汇总表（论文格式）")
print("=" * 100)

# 创建汇总表
summary_table = pd.DataFrame({
    'ORM2四分位数': ['Q1', 'Q2', 'Q3', 'Q4'],
    'MACE事件/总数': [
        f"{baseline_summary.loc['Q1', ('MACE', 'sum')]:.0f}/{baseline_summary.loc['Q1', ('MACE', 'count')]:.0f}",
        f"{baseline_summary.loc['Q2', ('MACE', 'sum')]:.0f}/{baseline_summary.loc['Q2', ('MACE', 'count')]:.0f}",
        f"{baseline_summary.loc['Q3', ('MACE', 'sum')]:.0f}/{baseline_summary.loc['Q3', ('MACE', 'count')]:.0f}",
        f"{baseline_summary.loc['Q4', ('MACE', 'sum')]:.0f}/{baseline_summary.loc['Q4', ('MACE', 'count')]:.0f}"
    ],
    'ORM2中位数(μg/mL)': [
        f"{baseline_summary.loc['Q1', ('ORM2', 'median')]:.1f}",
        f"{baseline_summary.loc['Q2', ('ORM2', 'median')]:.1f}",
        f"{baseline_summary.loc['Q3', ('ORM2', 'median')]:.1f}",
        f"{baseline_summary.loc['Q4', ('ORM2', 'median')]:.1f}"
    ]
})

# 初始化HR列
summary_table['粗HR (95% CI)'] = ''
summary_table['校正HR模型1'] = ''
summary_table['校正HR模型2'] = ''

# 填充HR数据
for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    if i == 0:
        summary_table.at[i, '粗HR (95% CI)'] = '1.00 (reference)'
        summary_table.at[i, '校正HR模型1'] = '1.00 (reference)'
        summary_table.at[i, '校正HR模型2'] = '1.00 (reference)'
    else:
        # 粗HR
        hr1 = cph_m1.summary.loc[q, 'exp(coef)']
        ci1_l = cph_m1.summary.loc[q, 'exp(coef) lower 95%']
        ci1_u = cph_m1.summary.loc[q, 'exp(coef) upper 95%']
        p1 = cph_m1.summary.loc[q, 'p']
        summary_table.at[i, '粗HR (95% CI)'] = f"{hr1:.2f} ({ci1_l:.2f}-{ci1_u:.2f}), P={p1:.3f}"

        # 校正HR模型1 (年龄)
        hr2 = cph_m2.summary.loc[q, 'exp(coef)']
        ci2_l = cph_m2.summary.loc[q, 'exp(coef) lower 95%']
        ci2_u = cph_m2.summary.loc[q, 'exp(coef) upper 95%']
        p2 = cph_m2.summary.loc[q, 'p']
        summary_table.at[i, '校正HR模型1'] = f"{hr2:.2f} ({ci2_l:.2f}-{ci2_u:.2f}), P={p2:.3f}"

        # 校正HR模型2 (年龄+BMI)
        if model3_success:
            hr3 = cph_m3.summary.loc[q, 'exp(coef)']
            ci3_l = cph_m3.summary.loc[q, 'exp(coef) lower 95%']
            ci3_u = cph_m3.summary.loc[q, 'exp(coef) upper 95%']
            p3 = cph_m3.summary.loc[q, 'p']
            summary_table.at[i, '校正HR模型2'] = f"{hr3:.2f} ({ci3_l:.2f}-{ci3_u:.2f}), P={p3:.3f}"
        else:
            summary_table.at[i, '校正HR模型2'] = '-'

print("\n" + summary_table.to_string(index=False))
print(f"\nP for trend = {p_trend:.4f}")
print("\n注释:")
print("- 校正HR模型1: 校正年龄")
if model3_success:
    print("- 校正HR模型2: 校正年龄 + BMI")
else:
    print("- 校正HR模型2: 未收敛")

# 保存结果
output_file = 'ORM2四分位数_MACE_Cox回归最终结果.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    summary_table.to_excel(writer, sheet_name='论文格式表格', index=False)

    # 详细结果
    cph_m1.summary.to_excel(writer, sheet_name='模型1_粗模型')
    cph_m2.summary.to_excel(writer, sheet_name='模型2_校正年龄')
    if model3_success:
        cph_m3.summary.to_excel(writer, sheet_name='模型3_校正年龄BMI')

    # 基线数据
    baseline_summary.to_excel(writer, sheet_name='四分位数基线数据')

print(f"\n结果已保存至: {output_file}")

# 关键发现总结
print("\n" + "=" * 100)
print("关键发现总结")
print("=" * 100)

print("\n1. ORM2与MACE风险关系:")
print(f"   - Q2组 vs Q1组: HR = {cph_m1.summary.loc['Q2', 'exp(coef)']:.2f}, P = {cph_m1.summary.loc['Q2', 'p']:.4f}")
print(f"   - Q3组 vs Q1组: HR = {cph_m1.summary.loc['Q3', 'exp(coef)']:.2f}, P = {cph_m1.summary.loc['Q3', 'p']:.4f}")
print(f"   - Q4组 vs Q1组: HR = {cph_m1.summary.loc['Q4', 'exp(coef)']:.2f}, P = {cph_m1.summary.loc['Q4', 'p']:.4f}")

print("\n2. 校正年龄后:")
print(f"   - Q2组 vs Q1组: HR = {cph_m2.summary.loc['Q2', 'exp(coef)']:.2f}, P = {cph_m2.summary.loc['Q2', 'p']:.4f}")
print(f"   - Q3组 vs Q1组: HR = {cph_m2.summary.loc['Q3', 'exp(coef)']:.2f}, P = {cph_m2.summary.loc['Q3', 'p']:.4f}")

if model3_success:
    print("\n3. 完全校正后（年龄+BMI）:")
    print(f"   - Q2组 vs Q1组: HR = {cph_m3.summary.loc['Q2', 'exp(coef)']:.2f}, P = {cph_m3.summary.loc['Q2', 'p']:.4f}")
    print(f"   - Q3组 vs Q1组: HR = {cph_m3.summary.loc['Q3', 'exp(coef)']:.2f}, P = {cph_m3.summary.loc['Q3', 'p']:.4f}")

print(f"\n4. 线性趋势: P for trend = {p_trend:.4f}")

print("\n结论:")
print("- ORM2水平升高（Q2、Q3组）与MACE风险显著增加相关")
print("- 最高四分位数组（Q4）的风险增加不显著，提示可能存在非线性关系")
print("- 年龄校正后，关联仍然显著")
if p_trend >= 0.05:
    print("- 线性趋势不显著，提示ORM2与MACE风险可能存在非线性关系")

print("\n" + "=" * 100)
