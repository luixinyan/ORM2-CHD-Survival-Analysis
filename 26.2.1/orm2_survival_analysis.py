#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
冠心病患者ORM2与结局事件关系的生存分析
分别分析：1) MACE  2) 全因死亡
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 读取数据
file_path = r'd:\onedrive\博士\博三\3-ORM2-第二次ELISA后开始最后分析\26.1.18\26.2.1\13.单独列出的冠心病.xlsx'
df = pd.read_excel(file_path)

print("=" * 100)
print("ORM2与冠心病患者结局事件的生存分析")
print("=" * 100)
print(f"\n总样本量: {len(df)} 例冠心病患者")
print(f"MACE事件: {df['MACE'].sum()} 例 ({df['MACE'].sum()/len(df)*100:.1f}%)")
print(f"全因死亡: {df['全因死亡'].sum()} 例 ({df['全因死亡'].sum()/len(df)*100:.1f}%)")

# 检查关键变量
print("\n" + "=" * 100)
print("关键变量检查")
print("=" * 100)
print(f"ORM2缺失值: {df['ORM2'].isnull().sum()}")
print(f"MACE缺失值: {df['MACE'].isnull().sum()}")
print(f"全因死亡缺失值: {df['全因死亡'].isnull().sum()}")
print(f"MACE_随访天数缺失值: {df['MACE_随访天数'].isnull().sum()}")
print(f"死亡_随访天数缺失值: {df['死亡_随访天数'].isnull().sum()}")

# ORM2四分位数分组
df['ORM2_quartile'] = pd.qcut(df['ORM2'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
quartile_cutoffs = df.groupby('ORM2_quartile')['ORM2'].agg(['min', 'max', 'median'])
print(f"\nORM2四分位数分组:")
print(quartile_cutoffs)

print("\n" + "=" * 100)
print("第一部分：ORM2与MACE的关系")
print("=" * 100)

# ========================================
# 1. MACE分析
# ========================================

# 1.1 描述性统计
print("\n【1.1 MACE事件发生率 - 按ORM2四分位数】")
mace_by_quartile = df.groupby('ORM2_quartile').agg({
    'MACE': ['sum', 'count', 'mean']
}).round(4)
mace_by_quartile.columns = ['事件数', '总数', '发生率']
print(mace_by_quartile)

# Chi-square test
from scipy.stats import chi2_contingency
ct_mace = pd.crosstab(df['ORM2_quartile'], df['MACE'])
chi2, p_value, dof, expected = chi2_contingency(ct_mace)
print(f"\nChi-square检验: chi2 = {chi2:.3f}, P = {p_value:.4f}")

# 1.2 Kaplan-Meier生存分析
print("\n【1.2 Kaplan-Meier生存分析 - MACE】")
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(10, 6))

for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    mask = df['ORM2_quartile'] == quartile
    kmf.fit(
        durations=df.loc[mask, 'MACE_随访天数'],
        event_observed=df.loc[mask, 'MACE'],
        label=f'{quartile}'
    )
    kmf.plot_survival_function(ax=ax, ci_show=True)

plt.xlabel('随访时间（天）', fontsize=12)
plt.ylabel('无MACE生存率', fontsize=12)
plt.title('Kaplan-Meier生存曲线：ORM2四分位数与MACE', fontsize=14, fontweight='bold')
plt.legend(title='ORM2分组', loc='lower left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('KM_curve_MACE_by_ORM2_quartile.png', dpi=300, bbox_inches='tight')
print(">> Kaplan-Meier曲线已保存: KM_curve_MACE_by_ORM2_quartile.png")

# Log-rank检验
print("\n【Log-rank检验 - 各组间比较】")
groups = df['ORM2_quartile'].unique()
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        g1, g2 = groups[i], groups[j]
        mask1 = df['ORM2_quartile'] == g1
        mask2 = df['ORM2_quartile'] == g2

        result = logrank_test(
            durations_A=df.loc[mask1, 'MACE_随访天数'],
            durations_B=df.loc[mask2, 'MACE_随访天数'],
            event_observed_A=df.loc[mask1, 'MACE'],
            event_observed_B=df.loc[mask2, 'MACE']
        )
        print(f"{g1} vs {g2}: chi2 = {result.test_statistic:.3f}, P = {result.p_value:.4f}")

# 1.3 Cox比例风险回归
print("\n【1.3 Cox比例风险回归 - MACE】")

# 准备数据
cox_data_mace = df[['ORM2', 'log10_ORM2', 'MACE_随访天数', 'MACE', '年龄', '性别', 'BMI']].copy()
cox_data_mace = cox_data_mace.dropna()
cox_data_mace.columns = ['ORM2', 'log10_ORM2', 'duration', 'event', '年龄', '性别', 'BMI']

# 单因素Cox回归 - ORM2连续变量
print("\n单因素Cox回归（ORM2连续变量）:")
cph = CoxPHFitter()
cph.fit(cox_data_mace[['duration', 'event', 'ORM2']], duration_col='duration', event_col='event')
print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']])
print(f"HR (95% CI): {cph.summary.loc['ORM2', 'exp(coef)']:.3f} "
      f"({cph.summary.loc['ORM2', 'exp(coef) lower 95%']:.3f}-"
      f"{cph.summary.loc['ORM2', 'exp(coef) upper 95%']:.3f})")

# 单因素Cox回归 - log10_ORM2
print("\n单因素Cox回归（log10_ORM2）:")
cph.fit(cox_data_mace[['duration', 'event', 'log10_ORM2']], duration_col='duration', event_col='event')
print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']])
print(f"HR (95% CI): {cph.summary.loc['log10_ORM2', 'exp(coef)']:.3f} "
      f"({cph.summary.loc['log10_ORM2', 'exp(coef) lower 95%']:.3f}-"
      f"{cph.summary.loc['log10_ORM2', 'exp(coef) upper 95%']:.3f})")

# 多因素Cox回归（校正年龄、性别、BMI）
print("\n多因素Cox回归（校正年龄、性别、BMI）:")
# 标准化连续变量以改善收敛性
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cox_data_mace_scaled = cox_data_mace.copy()
cox_data_mace_scaled[['ORM2', '年龄', 'BMI']] = scaler.fit_transform(cox_data_mace[['ORM2', '年龄', 'BMI']])

try:
    cph.fit(cox_data_mace_scaled[['duration', 'event', 'ORM2', '年龄', '性别', 'BMI']],
            duration_col='duration', event_col='event')
    print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']])
    print(f"\nORM2校正后HR (95% CI): {cph.summary.loc['ORM2', 'exp(coef)']:.3f} "
          f"({cph.summary.loc['ORM2', 'exp(coef) lower 95%']:.3f}-"
          f"{cph.summary.loc['ORM2', 'exp(coef) upper 95%']:.3f})")
    print("注意：此处ORM2已标准化，HR代表ORM2增加1个标准差的风险比")
except Exception as e:
    print(f"多因素Cox回归失败: {e}")
    print("尝试使用log10_ORM2...")
    try:
        cox_data_mace_log = cox_data_mace.copy()
        cox_data_mace_log[['log10_ORM2', '年龄', 'BMI']] = scaler.fit_transform(cox_data_mace[['log10_ORM2', '年龄', 'BMI']])
        cph.fit(cox_data_mace_log[['duration', 'event', 'log10_ORM2', '年龄', '性别', 'BMI']],
                duration_col='duration', event_col='event')
        print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']])
        print(f"\nlog10_ORM2校正后HR (95% CI): {cph.summary.loc['log10_ORM2', 'exp(coef)']:.3f} "
              f"({cph.summary.loc['log10_ORM2', 'exp(coef) lower 95%']:.3f}-"
              f"{cph.summary.loc['log10_ORM2', 'exp(coef) upper 95%']:.3f})")
    except Exception as e2:
        print(f"使用log10_ORM2也失败: {e2}")

# 1.4 ROC曲线分析
print("\n【1.4 ROC曲线分析 - MACE】")
fpr, tpr, thresholds = roc_curve(df['MACE'], df['ORM2'])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='参考线')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (1-特异度)', fontsize=12)
plt.ylabel('真阳性率 (敏感度)', fontsize=12)
plt.title('ROC曲线：ORM2预测MACE', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ROC_curve_ORM2_MACE.png', dpi=300, bbox_inches='tight')
print(f">> ROC曲线已保存: ROC_curve_ORM2_MACE.png")
print(f"AUC = {roc_auc:.3f}")

# 最佳截断值（Youden指数）
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
print(f"最佳截断值: {optimal_threshold:.2f}")
print(f"  敏感度: {tpr[optimal_idx]:.3f}")
print(f"  特异度: {1-fpr[optimal_idx]:.3f}")
print(f"  Youden指数: {youden_index[optimal_idx]:.3f}")


print("\n" + "=" * 100)
print("第二部分：ORM2与全因死亡的关系")
print("=" * 100)

# ========================================
# 2. 全因死亡分析
# ========================================

# 2.1 描述性统计
print("\n【2.1 全因死亡发生率 - 按ORM2四分位数】")
death_by_quartile = df.groupby('ORM2_quartile').agg({
    '全因死亡': ['sum', 'count', 'mean']
}).round(4)
death_by_quartile.columns = ['事件数', '总数', '发生率']
print(death_by_quartile)

# Chi-square test
ct_death = pd.crosstab(df['ORM2_quartile'], df['全因死亡'])
chi2, p_value, dof, expected = chi2_contingency(ct_death)
print(f"\nChi-square检验: chi2 = {chi2:.3f}, P = {p_value:.4f}")

# 2.2 Kaplan-Meier生存分析
print("\n【2.2 Kaplan-Meier生存分析 - 全因死亡】")
fig, ax = plt.subplots(figsize=(10, 6))

for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
    mask = df['ORM2_quartile'] == quartile
    kmf.fit(
        durations=df.loc[mask, '死亡_随访天数'],
        event_observed=df.loc[mask, '全因死亡'],
        label=f'{quartile}'
    )
    kmf.plot_survival_function(ax=ax, ci_show=True)

plt.xlabel('随访时间（天）', fontsize=12)
plt.ylabel('生存率', fontsize=12)
plt.title('Kaplan-Meier生存曲线：ORM2四分位数与全因死亡', fontsize=14, fontweight='bold')
plt.legend(title='ORM2分组', loc='lower left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('KM_curve_Death_by_ORM2_quartile.png', dpi=300, bbox_inches='tight')
print(">> Kaplan-Meier曲线已保存: KM_curve_Death_by_ORM2_quartile.png")

# Log-rank检验
print("\n【Log-rank检验 - 各组间比较】")
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        g1, g2 = groups[i], groups[j]
        mask1 = df['ORM2_quartile'] == g1
        mask2 = df['ORM2_quartile'] == g2

        result = logrank_test(
            durations_A=df.loc[mask1, '死亡_随访天数'],
            durations_B=df.loc[mask2, '死亡_随访天数'],
            event_observed_A=df.loc[mask1, '全因死亡'],
            event_observed_B=df.loc[mask2, '全因死亡']
        )
        print(f"{g1} vs {g2}: chi2 = {result.test_statistic:.3f}, P = {result.p_value:.4f}")

# 2.3 Cox比例风险回归
print("\n【2.3 Cox比例风险回归 - 全因死亡】")

# 准备数据
cox_data_death = df[['ORM2', 'log10_ORM2', '死亡_随访天数', '全因死亡', '年龄', '性别', 'BMI']].copy()
cox_data_death = cox_data_death.dropna()
cox_data_death.columns = ['ORM2', 'log10_ORM2', 'duration', 'event', '年龄', '性别', 'BMI']

# 单因素Cox回归 - ORM2连续变量
print("\n单因素Cox回归（ORM2连续变量）:")
cph = CoxPHFitter()
cph.fit(cox_data_death[['duration', 'event', 'ORM2']], duration_col='duration', event_col='event')
print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']])
print(f"HR (95% CI): {cph.summary.loc['ORM2', 'exp(coef)']:.3f} "
      f"({cph.summary.loc['ORM2', 'exp(coef) lower 95%']:.3f}-"
      f"{cph.summary.loc['ORM2', 'exp(coef) upper 95%']:.3f})")

# 单因素Cox回归 - log10_ORM2
print("\n单因素Cox回归（log10_ORM2）:")
cph.fit(cox_data_death[['duration', 'event', 'log10_ORM2']], duration_col='duration', event_col='event')
print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']])
print(f"HR (95% CI): {cph.summary.loc['log10_ORM2', 'exp(coef)']:.3f} "
      f"({cph.summary.loc['log10_ORM2', 'exp(coef) lower 95%']:.3f}-"
      f"{cph.summary.loc['log10_ORM2', 'exp(coef) upper 95%']:.3f})")

# 多因素Cox回归（校正年龄、性别、BMI）
print("\n多因素Cox回归（校正年龄、性别、BMI）:")
# 标准化连续变量以改善收敛性
cox_data_death_scaled = cox_data_death.copy()
cox_data_death_scaled[['ORM2', '年龄', 'BMI']] = scaler.fit_transform(cox_data_death[['ORM2', '年龄', 'BMI']])

try:
    cph.fit(cox_data_death_scaled[['duration', 'event', 'ORM2', '年龄', '性别', 'BMI']],
            duration_col='duration', event_col='event')
    print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']])
    print(f"\nORM2校正后HR (95% CI): {cph.summary.loc['ORM2', 'exp(coef)']:.3f} "
          f"({cph.summary.loc['ORM2', 'exp(coef) lower 95%']:.3f}-"
          f"{cph.summary.loc['ORM2', 'exp(coef) upper 95%']:.3f})")
    print("注意：此处ORM2已标准化，HR代表ORM2增加1个标准差的风险比")
except Exception as e:
    print(f"多因素Cox回归失败: {e}")
    print("尝试使用log10_ORM2...")
    try:
        cox_data_death_log = cox_data_death.copy()
        cox_data_death_log[['log10_ORM2', '年龄', 'BMI']] = scaler.fit_transform(cox_data_death[['log10_ORM2', '年龄', 'BMI']])
        cph.fit(cox_data_death_log[['duration', 'event', 'log10_ORM2', '年龄', '性别', 'BMI']],
                duration_col='duration', event_col='event')
        print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'z', 'p']])
        print(f"\nlog10_ORM2校正后HR (95% CI): {cph.summary.loc['log10_ORM2', 'exp(coef)']:.3f} "
              f"({cph.summary.loc['log10_ORM2', 'exp(coef) lower 95%']:.3f}-"
              f"{cph.summary.loc['log10_ORM2', 'exp(coef) upper 95%']:.3f})")
    except Exception as e2:
        print(f"使用log10_ORM2也失败: {e2}")

# 2.4 ROC曲线分析
print("\n【2.4 ROC曲线分析 - 全因死亡】")
fpr, tpr, thresholds = roc_curve(df['全因死亡'], df['ORM2'])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='参考线')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (1-特异度)', fontsize=12)
plt.ylabel('真阳性率 (敏感度)', fontsize=12)
plt.title('ROC曲线：ORM2预测全因死亡', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ROC_curve_ORM2_Death.png', dpi=300, bbox_inches='tight')
print(f">> ROC曲线已保存: ROC_curve_ORM2_Death.png")
print(f"AUC = {roc_auc:.3f}")

# 最佳截断值
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
print(f"最佳截断值: {optimal_threshold:.2f}")
print(f"  敏感度: {tpr[optimal_idx]:.3f}")
print(f"  特异度: {1-fpr[optimal_idx]:.3f}")
print(f"  Youden指数: {youden_index[optimal_idx]:.3f}")

print("\n" + "=" * 100)
print("分析完成！")
print("=" * 100)
print("\n生成的文件:")
print("1. KM_curve_MACE_by_ORM2_quartile.png - MACE的Kaplan-Meier生存曲线")
print("2. ROC_curve_ORM2_MACE.png - ORM2预测MACE的ROC曲线")
print("3. KM_curve_Death_by_ORM2_quartile.png - 全因死亡的Kaplan-Meier生存曲线")
print("4. ROC_curve_ORM2_Death.png - ORM2预测全因死亡的ROC曲线")
