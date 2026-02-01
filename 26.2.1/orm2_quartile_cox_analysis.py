#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ORM2四分位数与MACE事件的Cox回归分析
包括单因素和多因素分析
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import warnings
warnings.filterwarnings('ignore')

# 读取数据
file_path = r'd:\onedrive\博士\博三\3-ORM2-第二次ELISA后开始最后分析\26.1.18\26.2.1\13.单独列出的冠心病.xlsx'
df = pd.read_excel(file_path)

print("=" * 100)
print("ORM2四分位数与MACE事件的Cox回归分析")
print("=" * 100)

# ORM2四分位数分组
df['ORM2_quartile'] = pd.qcut(df['ORM2'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 查看各组基线信息
print("\n【基线信息】")
print(f"总样本量: {len(df)} 例")
print(f"MACE事件: {df['MACE'].sum()} 例 ({df['MACE'].sum()/len(df)*100:.1f}%)")

quartile_summary = df.groupby('ORM2_quartile').agg({
    'ORM2': ['min', 'max', 'median'],
    'MACE': ['sum', 'count']
})
quartile_summary.columns = ['ORM2_最小值', 'ORM2_最大值', 'ORM2_中位数', 'MACE事件数', '总人数']
quartile_summary['MACE发生率(%)'] = (quartile_summary['MACE事件数'] / quartile_summary['总人数'] * 100).round(2)
print("\n各四分位数组基线信息:")
print(quartile_summary)

# 准备Cox回归数据
print("\n" + "=" * 100)
print("准备Cox回归数据")
print("=" * 100)

# 创建哑变量（以Q1为参考组）
df_cox = df[['ORM2_quartile', 'MACE_随访天数', 'MACE', '年龄', '性别', 'BMI']].copy()
df_cox = df_cox.dropna()

print(f"\n完整数据集: {len(df_cox)} 例")
print(f"MACE事件: {df_cox['MACE'].sum()} 例")

# 创建哑变量（Q1为参考组）
df_cox = pd.get_dummies(df_cox, columns=['ORM2_quartile'], drop_first=True, dtype=int)
print("\n哑变量编码完成（Q1作为参考组）:")
print(df_cox.columns.tolist())

# 重命名列以便于理解
df_cox.columns = ['duration', 'event', '年龄', '性别', 'BMI', 'Q2', 'Q3', 'Q4']

print("\n" + "=" * 100)
print("单因素Cox回归分析（ORM2四分位数）")
print("=" * 100)

# 单因素Cox回归
cph = CoxPHFitter()
cph.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4']],
        duration_col='duration',
        event_col='event')

print("\n单因素Cox回归结果:")
print(cph.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'se(coef)', 'z', 'p']])

print("\n各组相对于Q1的风险比(HR):")
print(f"Q1 (参考组): HR = 1.00")
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph.summary.loc[q, 'exp(coef)']
    ci_lower = cph.summary.loc[q, 'exp(coef) lower 95%']
    ci_upper = cph.summary.loc[q, 'exp(coef) upper 95%']
    p_value = cph.summary.loc[q, 'p']
    print(f"{q}: HR = {hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), P = {p_value:.4f}")

# 趋势检验（将四分位数作为连续变量0,1,2,3）
print("\n" + "=" * 100)
print("趋势检验（P for trend）")
print("=" * 100)

df_trend = df[['MACE_随访天数', 'MACE', 'ORM2_quartile']].copy()
df_trend = df_trend.dropna()
df_trend['quartile_numeric'] = df_trend['ORM2_quartile'].map({'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3})
df_trend.columns = ['duration', 'event', 'quartile_cat', 'quartile_numeric']

cph_trend = CoxPHFitter()
cph_trend.fit(df_trend[['duration', 'event', 'quartile_numeric']],
              duration_col='duration',
              event_col='event')

p_trend = cph_trend.summary.loc['quartile_numeric', 'p']
print(f"\nP for trend = {p_trend:.4f}")
if p_trend < 0.05:
    print("结论: ORM2四分位数与MACE风险存在显著线性趋势")
else:
    print("结论: ORM2四分位数与MACE风险未显示显著线性趋势")

print("\n" + "=" * 100)
print("多因素Cox回归分析（校正年龄、性别、BMI）")
print("=" * 100)

# 模型1: 未校正（粗模型）
print("\n【模型1: 粗模型（未校正）】")
cph_m1 = CoxPHFitter()
cph_m1.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4']],
           duration_col='duration',
           event_col='event')
print(cph_m1.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

# 模型2: 校正年龄、性别
print("\n【模型2: 校正年龄、性别】")
try:
    cph_m2 = CoxPHFitter()
    cph_m2.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄', '性别']],
               duration_col='duration',
               event_col='event')
    print(cph_m2.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])
except Exception as e:
    print(f"模型2拟合失败: {e}")

# 模型3: 校正年龄、性别、BMI
print("\n【模型3: 校正年龄、性别、BMI（完全校正）】")
try:
    cph_m3 = CoxPHFitter()
    cph_m3.fit(df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄', '性别', 'BMI']],
               duration_col='duration',
               event_col='event')
    print(cph_m3.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

    print("\n完全校正模型中各组HR总结:")
    print(f"Q1 (参考组): HR = 1.00")
    for q in ['Q2', 'Q3', 'Q4']:
        hr = cph_m3.summary.loc[q, 'exp(coef)']
        ci_lower = cph_m3.summary.loc[q, 'exp(coef) lower 95%']
        ci_upper = cph_m3.summary.loc[q, 'exp(coef) upper 95%']
        p_value = cph_m3.summary.loc[q, 'p']
        print(f"{q}: HR = {hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), P = {p_value:.4f}")

    # 检验比例风险假设
    print("\n" + "=" * 100)
    print("比例风险假设检验（Proportional Hazard Test）")
    print("=" * 100)
    try:
        ph_test = proportional_hazard_test(cph_m3, df_cox[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄', '性别', 'BMI']],
                                          time_transform='rank')
        print(ph_test)
        print("\n注: P值>0.05表明满足比例风险假设")
    except Exception as e:
        print(f"比例风险假设检验失败: {e}")

except Exception as e:
    print(f"模型3拟合失败: {e}")
    print("\n尝试使用标准化变量...")
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_cox_scaled = df_cox.copy()
        df_cox_scaled[['年龄', 'BMI']] = scaler.fit_transform(df_cox[['年龄', 'BMI']])

        cph_m3_scaled = CoxPHFitter()
        cph_m3_scaled.fit(df_cox_scaled[['duration', 'event', 'Q2', 'Q3', 'Q4', '年龄', '性别', 'BMI']],
                         duration_col='duration',
                         event_col='event')
        print("\n使用标准化变量的结果:")
        print(cph_m3_scaled.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

        print("\n完全校正模型中各组HR总结:")
        print(f"Q1 (参考组): HR = 1.00")
        for q in ['Q2', 'Q3', 'Q4']:
            hr = cph_m3_scaled.summary.loc[q, 'exp(coef)']
            ci_lower = cph_m3_scaled.summary.loc[q, 'exp(coef) lower 95%']
            ci_upper = cph_m3_scaled.summary.loc[q, 'exp(coef) upper 95%']
            p_value = cph_m3_scaled.summary.loc[q, 'p']
            print(f"{q}: HR = {hr:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f}), P = {p_value:.4f}")
    except Exception as e2:
        print(f"标准化后仍然失败: {e2}")

# 生成结果表格
print("\n" + "=" * 100)
print("结果汇总表格（适合论文发表）")
print("=" * 100)

# 创建汇总表
results_table = pd.DataFrame({
    'ORM2分组': ['Q1', 'Q2', 'Q3', 'Q4'],
    '事件数/总数': [
        f"{quartile_summary.loc['Q1', 'MACE事件数']:.0f}/{quartile_summary.loc['Q1', '总人数']:.0f}",
        f"{quartile_summary.loc['Q2', 'MACE事件数']:.0f}/{quartile_summary.loc['Q2', '总人数']:.0f}",
        f"{quartile_summary.loc['Q3', 'MACE事件数']:.0f}/{quartile_summary.loc['Q3', '总人数']:.0f}",
        f"{quartile_summary.loc['Q4', 'MACE事件数']:.0f}/{quartile_summary.loc['Q4', '总人数']:.0f}"
    ],
    '发生率(%)': [
        f"{quartile_summary.loc['Q1', 'MACE发生率(%)']:.2f}",
        f"{quartile_summary.loc['Q2', 'MACE发生率(%)']:.2f}",
        f"{quartile_summary.loc['Q3', 'MACE发生率(%)']:.2f}",
        f"{quartile_summary.loc['Q4', 'MACE发生率(%)']:.2f}"
    ]
})

# 添加单因素HR
hr_crude = ['1.00 (reference)']
for q in ['Q2', 'Q3', 'Q4']:
    hr = cph_m1.summary.loc[q, 'exp(coef)']
    ci_lower = cph_m1.summary.loc[q, 'exp(coef) lower 95%']
    ci_upper = cph_m1.summary.loc[q, 'exp(coef) upper 95%']
    p_value = cph_m1.summary.loc[q, 'p']
    hr_crude.append(f"{hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), P={p_value:.3f}")
results_table['粗HR (95% CI)'] = hr_crude

# 如果模型3成功，添加校正后HR
try:
    hr_adjusted = ['1.00 (reference)']
    for q in ['Q2', 'Q3', 'Q4']:
        hr = cph_m3.summary.loc[q, 'exp(coef)']
        ci_lower = cph_m3.summary.loc[q, 'exp(coef) lower 95%']
        ci_upper = cph_m3.summary.loc[q, 'exp(coef) upper 95%']
        p_value = cph_m3.summary.loc[q, 'p']
        hr_adjusted.append(f"{hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), P={p_value:.3f}")
    results_table['校正HR (95% CI)'] = hr_adjusted
    print("\n注: 校正HR已校正年龄、性别、BMI")
except:
    try:
        hr_adjusted = ['1.00 (reference)']
        for q in ['Q2', 'Q3', 'Q4']:
            hr = cph_m3_scaled.summary.loc[q, 'exp(coef)']
            ci_lower = cph_m3_scaled.summary.loc[q, 'exp(coef) lower 95%']
            ci_upper = cph_m3_scaled.summary.loc[q, 'exp(coef) upper 95%']
            p_value = cph_m3_scaled.summary.loc[q, 'p']
            hr_adjusted.append(f"{hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f}), P={p_value:.3f}")
        results_table['校正HR (95% CI)'] = hr_adjusted
        print("\n注: 校正HR已校正年龄、性别、BMI（使用标准化变量）")
    except:
        results_table['校正HR (95% CI)'] = ['多因素模型未收敛'] * 4

print("\n" + results_table.to_string(index=False))

# 保存结果到Excel
output_file = 'ORM2四分位数_MACE_Cox回归结果.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_table.to_excel(writer, sheet_name='汇总结果', index=False)
    quartile_summary.to_excel(writer, sheet_name='四分位数基线')

    # 保存详细的Cox回归结果
    cph_m1.summary.to_excel(writer, sheet_name='粗模型详细结果')
    try:
        cph_m3.summary.to_excel(writer, sheet_name='校正模型详细结果')
    except:
        try:
            cph_m3_scaled.summary.to_excel(writer, sheet_name='校正模型详细结果')
        except:
            pass

print(f"\n结果已保存至: {output_file}")

print("\n" + "=" * 100)
print("分析完成！")
print("=" * 100)
print("\n主要发现:")
print("1. 单因素Cox回归显示ORM2四分位数与MACE风险的关系")
print("2. 多因素Cox回归校正了年龄、性别、BMI等混杂因素")
print(f"3. 趋势检验P值 = {p_trend:.4f}")
print("\n建议:")
print("- 查看生成的Excel文件获取完整结果")
print("- 如需进一步分析，可以增加其他协变量（如糖尿病、高血压等）")
print("- 可以进一步进行亚组分析或敏感性分析")
