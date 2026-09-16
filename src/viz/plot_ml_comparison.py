# -*- coding: utf-8 -*-
"""
图4: 与传统机器学习模型对比
学术风格分组柱状图：6个模型 × Top-1/Macro-F1 指标
"""
import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

# ====== 数据 ======
models = ['CatBoost\n(本文)', 'XGBoost', 'LightGBM', 'Random\nForest', 'Logistic\nRegression', 'MLP']

top1_mean = [42.5, 41.6, 41.3, 41.1, 30.8, 26.3]
top1_std  = [0.9,  1.6,  0.9,  1.5,  0.5,  0.2]

top5_mean = [88.6, 87.4, 87.0, 87.5, 77.4, 73.0]
top5_std  = [0.4,  1.0,  1.1,  0.6,  1.0,  2.8]

macro_f1_mean = [30.6, 29.3, 28.5, 25.5, 5.1, 2.9]
macro_f1_std  = [2.2,  2.5,  2.5,  3.5,  0.6,  0.2]

mrr_mean = [0.608, 0.600, 0.597, 0.596, 0.501, 0.465]
mrr_std  = [0.006, 0.009, 0.007, 0.010, 0.004, 0.007]

# ====== 配色 (TCM 绿+金) ======
GREEN = '#1a5632'
GREEN_MID = '#2d8a4e'
GREEN_LIGHT = '#5ba872'
GREEN_PALE = '#a8d5ba'
GOLD = '#c8960c'
GRAY = '#b0b0b0'
GRAY_LIGHT = '#d5d5d5'

# ====== 图4: Top-1 + Macro-F1 分组柱状图 ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
fig.patch.set_facecolor('#ffffff')

x = np.arange(len(models))
width = 0.55

# ---- (a) Top-1 准确率 ----
colors_top1 = [GOLD if i == 0 else GREEN_MID if i == 1 else GREEN_LIGHT
               if i == 2 else GREEN_PALE if i == 3 else GRAY_LIGHT for i in range(6)]
bars1 = ax1.bar(x, top1_mean, width, color=colors_top1, edgecolor='white', linewidth=0.8, zorder=3)
ax1.errorbar(x, top1_mean, yerr=top1_std, fmt='none', ecolor='#555555',
             capsize=4, capthick=0.8, elinewidth=0.8, zorder=4)

# 数值标注
for i, (bar, val) in enumerate(zip(bars1, top1_mean)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5,
             color='#333333', fontweight='normal')

ax1.set_ylabel('Top-1 准确率 (%)', fontsize=11, color='#333333')
ax1.set_title('(a) Top-1 准确率对比', fontsize=12, fontweight='bold', color='#222222', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=8.5, color='#333333')
ax1.set_ylim(0, 55)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax1.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#cccccc')
ax1.spines['bottom'].set_color('#cccccc')
ax1.tick_params(colors='#666666', labelsize=8.5)

# ---- (b) Macro-F1 ----
colors_mf1 = [GOLD if i == 0 else GREEN_MID if i == 1 else GREEN_LIGHT
              if i == 2 else GREEN_PALE if i == 3 else GRAY_LIGHT for i in range(6)]
bars2 = ax2.bar(x, macro_f1_mean, width, color=colors_mf1, edgecolor='white', linewidth=0.8, zorder=3)
ax2.errorbar(x, macro_f1_mean, yerr=macro_f1_std, fmt='none', ecolor='#555555',
             capsize=4, capthick=0.8, elinewidth=0.8, zorder=4)

for i, (bar, val) in enumerate(zip(bars2, macro_f1_mean)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5,
             color='#333333', fontweight='normal')

ax2.set_ylabel('Macro-F1 (%)', fontsize=11, color='#333333')
ax2.set_title('(b) Macro-F1 对比', fontsize=12, fontweight='bold', color='#222222', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=8.5, color='#333333')
ax2.set_ylim(0, 40)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax2.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#cccccc')
ax2.spines['bottom'].set_color('#cccccc')
ax2.tick_params(colors='#666666', labelsize=8.5)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=GOLD, label='CatBoost (本文方法)'),
    Patch(facecolor=GREEN_MID, label='GBDT 类模型'),
    Patch(facecolor=GREEN_PALE, label='传统集成方法'),
    Patch(facecolor=GRAY_LIGHT, label='传统/深度学习'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=4,
           fontsize=8, frameon=True, framealpha=0.9, edgecolor='#dddddd',
           bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.93])
plt.savefig('outputs/figures/fig4_ml_comparison.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('fig4_ml_comparison.png saved')
plt.close()

# ====== 图5: 完整指标雷达图/汇总热力图 (Top-1, Top-3, Top-5, MRR, Micro-F1, Macro-F1) ======
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
fig.patch.set_facecolor('#ffffff')

# 归一化数据用于对比
metrics_names = ['Top-1', 'Top-3', 'Top-5', 'MRR', 'Micro-F1', 'Macro-F1']
data = {
    'CatBoost (本文)': [42.5, 73.5, 88.6, 60.8, 42.5, 30.6],
    'XGBoost':         [41.6, 72.6, 87.4, 60.0, 41.6, 29.3],
    'LightGBM':        [41.3, 71.5, 87.0, 59.7, 41.3, 28.5],
    'RandomForest':    [41.1, 71.7, 87.5, 59.6, 41.1, 25.5],
    'LogisticReg.':    [30.8, 60.6, 77.4, 50.1, 30.8,  5.1],
    'MLP':             [26.3, 55.6, 73.0, 46.5, 26.3,  2.9],
}

y_pos = np.arange(len(metrics_names))
bar_height = 0.12
colors_bar = [GOLD, GREEN_MID, GREEN_LIGHT, GREEN_PALE, GRAY_LIGHT, '#e0e0e0']

for i, (name, values) in enumerate(data.items()):
    offset = (i - 2.5) * bar_height
    ax.barh(y_pos + offset, values, bar_height, label=name,
            color=colors_bar[i], edgecolor='white', linewidth=0.5, zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(metrics_names, fontsize=10, color='#333333')
ax.set_xlabel('Score (%)', fontsize=11, color='#333333')
ax.set_title('多模型 × 多指标全面对比', fontsize=13, fontweight='bold', color='#222222', pad=12)
ax.set_xlim(0, 100)
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.grid(True, axis='x', alpha=0.25, linewidth=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.tick_params(colors='#666666', labelsize=9)
ax.legend(loc='lower right', fontsize=8, framealpha=0.9, edgecolor='#dddddd', ncol=2)

plt.tight_layout(pad=2.0)
plt.savefig('outputs/figures/fig5_ml_all_metrics.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('fig5_ml_all_metrics.png saved')
plt.close()

# ====== 数值输出 ======
print('\n数值汇总:')
for name, values in data.items():
    print(f'{name}: Top-1={values[0]:.1f}%, Top-3={values[1]:.1f}%, Top-5={values[2]:.1f}%, '
          f'MRR={values[3]:.1f}%, Micro-F1={values[4]:.1f}%, Macro-F1={values[5]:.1f}%')

print('\nDone!')