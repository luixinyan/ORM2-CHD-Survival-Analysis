#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ORM2四分位数与MACE - 多变量Cox回归分析
调整协变量：年龄、BMI、高血压、糖尿病、高脂血症
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 读取数据
file_path = r'13.单独列出的冠心病.xlsx'
df = pd.read_excel(file_path)

print("=" * 100)
print("ORM2四分位数与MACE - 多变量Cox回归分析")
print("调整协变量：年龄、BMI、高血压、糖尿病、高脂血症")
print("=" * 100)

# ORM2四分位数分组
df['ORM2_quartile'] = pd.qcut(df['ORM2'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 选择分析变量（不包括eGFR）
analysis_vars = ['ORM2', 'ORM2_quartile', 'MACE_随访天数', 'MACE',
                 '年龄', 'BMI', '高血压', '糖尿病', '高脂血症']

# 准备数据
df_analysis = df[analysis_vars].copy()

print("\n【数据准备】")
print("\n原始数据:")
print(f"总样本量: {len(df_analysis)} 例")
print(f"MACE事件: {df_analysis['MACE'].sum()} 例 ({df_analysis['MACE'].sum()/len(df_analysis)*100:.1f}%)")

print("\n各变量缺失情况:")
for var in analysis_vars[3:]:  # 跳过ORM2、分组和结局变量
    missing = df_analysis[var].isnull().sum()
    pct = missing/len(df_analysis)*100
    print(f"  {var}: {missing} ({pct:.1f}%)")

# 删除缺失值（完整病例分析）
n_before = len(df_analysis)
mace_before = df_analysis['MACE'].sum()
df_analysis = df_analysis.dropna()
n_after = len(df_analysis)
mace_after = df_analysis['MACE'].sum()

print(f"\n完整病例分析样本量: {n_after} 例")
print(f"  MACE事件: {mace_after} 例 ({mace_after/n_after*100:.1f}%)")
print(f"  排除样本: {n_before - n_after} 例 (排除{mace_before - mace_after}例MACE事件)")

# 基线特征（按ORM2四分位数）
print("\n" + "=" * 100)
print("基线特征（按ORM2四分位数）")
print("=" * 100)

baseline = df_analysis.groupby('ORM2_quartile').agg({
    'MACE': ['sum', 'count', 'mean'],
    'ORM2': ['median', 'min', 'max'],
    '年龄': ['mean', 'std'],
    'BMI': ['mean', 'std'],
    '高血压': ['sum', 'mean'],
    '糖尿病': ['sum', 'mean'],
    '高脂血症': ['sum', 'mean']
}).round(2)

print("\n各组基线特征:")
print(baseline)

# 各组MACE发生率
print("\n各组MACE发生率:")
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    n_total = baseline.loc[q, ('MACE', 'count')]
    n_mace = baseline.loc[q, ('MACE', 'sum')]
    rate = baseline.loc[q, ('MACE', 'mean')] * 100
    orm2_median = baseline.loc[q, ('ORM2', 'median')]
    print(f"  {q}: {n_mace:.0f}/{n_total:.0f} ({rate:.1f}%), ORM2中位数={orm2_median:.1f}")

# 创建分析数据集
df_cox = df_analysis.copy()

# 创建ORM2四分位数的哑变量（Q1为参考组）
df_cox = pd.get_dummies(df_cox, columns=['ORM2_quartile'], drop_first=True, dtype=int)

# 重命名列
df_cox.columns = ['ORM2_value', 'duration', 'event', '年龄', 'BMI', '高血压', '糖尿病', '高脂血症', 'Q2', 'Q3', 'Q4']

# 标准化连续变量
scaler = StandardScaler()
df_cox[['年龄_std', 'BMI_std']] = scaler.fit_transform(df_cox[['年龄', 'BMI']])

print("\n" + "=" * 100)
print("Cox回归分析 - 递进模型")
print("=" * 100)

# 存储结果
all_results = []

# ============================================================================
# 模型1: 粗模型（未校正）
# ============================================================================
print("\n【模型1: 粗模型（未校正）】")
print("变量: ORM2四分位数")

cph_m1 = CoxPHFitter()
cph_m1.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4']],
           duration_col='duration', event_col='event')

print("\nHR (95% CI):")
all_results.append({'模型': '模型1: 粗模型', '分组': 'Q1', 'HR': '1.00', 'CI': 'reference', 'P': '-'})
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m1.summary.loc[q, 'exp(coef)']
    ci_l = cph_m1.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m1.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m1.summary.loc[q, 'p']
    sig = "**" if p < 0.05 else "△" if p < 0.10 else ""
    print(f"  {q}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f} {sig}")
    all_results.append({'模型': '模型1: 粗模型', '分组': q, 'HR': f"{hr:.2f}",
                       'CI': f"{ci_l:.2f}-{ci_u:.2f}", 'P': f"{p:.4f}"})

# ============================================================================
# 模型2: 校正年龄 + BMI
# ============================================================================
print("\n【模型2: 校正年龄 + BMI】")
print("变量: ORM2四分位数 + 年龄 + BMI")

cph_m2 = CoxPHFitter()
cph_m2.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄_std', 'BMI_std']],
           duration_col='duration', event_col='event')

print("\nORM2四分位数 HR (95% CI):")
all_results.append({'模型': '模型2: 校正年龄+BMI', '分组': 'Q1', 'HR': '1.00', 'CI': 'reference', 'P': '-'})
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m2.summary.loc[q, 'exp(coef)']
    ci_l = cph_m2.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m2.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m2.summary.loc[q, 'p']
    sig = "**" if p < 0.05 else "△" if p < 0.10 else ""
    print(f"  {q}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f} {sig}")
    all_results.append({'模型': '模型2: 校正年龄+BMI', '分组': q, 'HR': f"{hr:.2f}",
                       'CI': f"{ci_l:.2f}-{ci_u:.2f}", 'P': f"{p:.4f}"})

print("\n协变量:")
for var, name in [('年龄_std', '年龄(每增加1 SD)'), ('BMI_std', 'BMI(每增加1 SD)')]:
    hr = cph_m2.summary.loc[var, 'exp(coef)']
    ci_l = cph_m2.summary.loc[var, 'exp(coef) lower 95%']
    ci_u = cph_m2.summary.loc[var, 'exp(coef) upper 95%']
    p = cph_m2.summary.loc[var, 'p']
    sig = "**" if p < 0.05 else ""
    print(f"  {name}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f} {sig}")

# ============================================================================
# 模型3: 校正年龄 + BMI + 心血管危险因素
# ============================================================================
print("\n【模型3: 校正年龄 + BMI + 高血压 + 糖尿病 + 高脂血症（完全校正模型）】")
print("变量: ORM2四分位数 + 年龄 + BMI + 高血压 + 糖尿病 + 高脂血症")

cph_m3 = CoxPHFitter(penalizer=0.01)
cph_m3.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4',
                   '年龄_std', 'BMI_std', '高血压', '糖尿病', '高脂血症']],
           duration_col='duration', event_col='event')

print("\nORM2四分位数 HR (95% CI):")
all_results.append({'模型': '模型3: 完全校正', '分组': 'Q1', 'HR': '1.00', 'CI': 'reference', 'P': '-'})
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m3.summary.loc[q, 'exp(coef)']
    ci_l = cph_m3.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m3.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m3.summary.loc[q, 'p']
    sig = "**" if p < 0.05 else "△" if p < 0.10 else ""
    print(f"  {q}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f} {sig}")
    all_results.append({'模型': '模型3: 完全校正', '分组': q, 'HR': f"{hr:.2f}",
                       'CI': f"{ci_l:.2f}-{ci_u:.2f}", 'P': f"{p:.4f}"})

print("\n协变量:")
cov_info = {
    '年龄_std': '年龄(每增加1 SD)',
    'BMI_std': 'BMI(每增加1 SD)',
    '高血压': '高血压(有vs无)',
    '糖尿病': '糖尿病(有vs无)',
    '高脂血症': '高脂血症(有vs无)'
}
for var, name in cov_info.items():
    hr = cph_m3.summary.loc[var, 'exp(coef)']
    ci_l = cph_m3.summary.loc[var, 'exp(coef) lower 95%']
    ci_u = cph_m3.summary.loc[var, 'exp(coef) upper 95%']
    p = cph_m3.summary.loc[var, 'p']
    sig = "**" if p < 0.05 else ""
    print(f"  {name}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f} {sig}")

# ============================================================================
# 生成汇总表格
# ============================================================================
print("\n" + "=" * 100)
print("结果汇总表（论文格式）")
print("=" * 100)

# 创建pivot表
pivot_data = []
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    row = {'ORM2分组': q}
    for model_name in ['模型1: 粗模型', '模型2: 校正年龄+BMI', '模型3: 完全校正']:
        subset = [r for r in all_results if r['模型'] == model_name and r['分组'] == q]
        if subset:
            r = subset[0]
            if r['HR'] == '1.00':
                row[model_name] = '1.00 (reference)'
            else:
                p_float = float(r['P'])
                p_str = f"P={p_float:.3f}" if p_float >= 0.001 else "P<0.001"
                sig = " **" if p_float < 0.05 else " △" if p_float < 0.10 else ""
                row[model_name] = f"{r['HR']} ({r['CI']}), {p_str}{sig}"
    pivot_data.append(row)

final_table = pd.DataFrame(pivot_data)
print("\n" + final_table.to_string(index=False))

print("\n注：** P<0.05（显著），△ P<0.10（边界显著）")

# 保存结果
output_file = 'ORM2四分位数_MACE_多变量Cox回归_最终版.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 汇总表
    final_table.to_excel(writer, sheet_name='论文格式表格', index=False)

    # 各模型详细结果
    cph_m1.summary.to_excel(writer, sheet_name='模型1_粗模型')
    cph_m2.summary.to_excel(writer, sheet_name='模型2_校正年龄BMI')
    cph_m3.summary.to_excel(writer, sheet_name='模型3_完全校正')

    # 基线特征
    baseline.to_excel(writer, sheet_name='基线特征')

    # 创建详细结果表
    results_df = pd.DataFrame(all_results)
    results_df.to_excel(writer, sheet_name='详细结果', index=False)

print(f"\n结果已保存至: {output_file}")

# ============================================================================
# 关键发现总结
# ============================================================================
print("\n" + "=" * 100)
print("关键发现总结")
print("=" * 100)

print(f"\n1. 样本量与事件数:")
print(f"   完整病例分析: {n_after} 例")
print(f"   MACE事件: {mace_after} 例 ({mace_after/n_after*100:.1f}%)")

print("\n2. ORM2四分位数与MACE风险:")

print("\n   【粗模型】:")
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m1.summary.loc[q, 'exp(coef)']
    ci_l = cph_m1.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m1.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m1.summary.loc[q, 'p']
    if p < 0.05:
        sig_text = "（显著）"
    elif p < 0.10:
        sig_text = "（边界显著）"
    else:
        sig_text = ""
    print(f"   {q} vs Q1: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f} {sig_text}")

print("\n   【完全校正模型】（校正年龄、BMI、高血压、糖尿病、高脂血症）:")
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m3.summary.loc[q, 'exp(coef)']
    ci_l = cph_m3.summary.loc[q, 'exp(coef) lower 95%']
    ci_u = cph_m3.summary.loc[q, 'exp(coef) upper 95%']
    p = cph_m3.summary.loc[q, 'p']
    if p < 0.05:
        sig_text = "（显著）"
    elif p < 0.10:
        sig_text = "（边界显著）"
    else:
        sig_text = ""
    print(f"   {q} vs Q1: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f} {sig_text}")

print("\n3. 协变量的独立作用（完全校正模型）:")
sig_covariates = []
for var, name in cov_info.items():
    p = cph_m3.summary.loc[var, 'p']
    if p < 0.05:
        hr = cph_m3.summary.loc[var, 'exp(coef)']
        ci_l = cph_m3.summary.loc[var, 'exp(coef) lower 95%']
        ci_u = cph_m3.summary.loc[var, 'exp(coef) upper 95%']
        sig_covariates.append(f"   {name}: HR = {hr:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f}), P = {p:.4f}")

if sig_covariates:
    print("   独立危险因素:")
    for item in sig_covariates:
        print(item)
else:
    print("   无协变量达到统计学显著性")

# 趋势检验
print("\n4. 线性趋势检验:")
df_trend = df_analysis[['MACE_随访天数', 'MACE', 'ORM2_quartile']].copy()
df_trend['quartile_numeric'] = df_trend['ORM2_quartile'].map({'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3})
cph_trend = CoxPHFitter()
cph_trend.fit(df_trend[['MACE_随访天数', 'MACE', 'quartile_numeric']].rename(
    columns={'MACE_随访天数': 'duration', 'MACE': 'event'}),
    duration_col='duration', event_col='event')
p_trend = cph_trend.summary.loc['quartile_numeric', 'p']
print(f"   P for trend = {p_trend:.4f}")
if p_trend < 0.05:
    print("   结论: ORM2水平与MACE风险存在显著的线性趋势")
else:
    print("   结论: 未观察到显著的线性趋势，提示可能存在非线性关系")

print("\n" + "=" * 100)
print("分析完成！")
print("=" * 100)
