# -*- coding: utf-8 -*-
"""绘制 CatBoost 训练 Loss 曲线 + 特征重要性 Top-15"""
import sys, os, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from catboost import CatBoostClassifier

# 中文字体配置 — 优先用系统自带、字重完整的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'normal'

# ====== 加载模型 ======
model = CatBoostClassifier()
model.load_model('web_app/catboost_model.cbm')
with open('web_app/feature_cols.pkl', 'rb') as f:
    feat_cols = pickle.load(f)

# ====== 数据准备 ======
loss = model.get_evals_result()['learn']['MultiClass']
fi = model.get_feature_importance()
feat_imp = sorted(zip(feat_cols, fi), key=lambda x: -x[1])[:15]
feat_names = [f[0] for f in feat_imp][::-1]
feat_values = [f[1] for f in feat_imp][::-1]

# 中文映射
NAME_MAP = {
    'meridian_心_ratio': '归心经', 'meridian_肺_ratio': '归肺经',
    'meridian_肾_ratio': '归肾经', 'meridian_脾_ratio': '归脾经',
    'meridian_胃_ratio': '归胃经', 'meridian_肝_ratio': '归肝经',
    'nature_温_ratio': '温性药材', 'nature_平_ratio': '平性药材',
    'volatile_ratio': '含挥发油', 'taste_甘_ratio': '甘味药材',
    'taste_苦_ratio': '苦味药材', 'total_weight': '总剂量',
    'min_dose': '最小剂量', 'max_min_ratio': '最大/最小剂量比',
    'log_total_weight': '对数总剂量',
}
feat_names_cn = [NAME_MAP.get(n, n) for n in feat_names]

# ====== 配色（TCM 主题：绿 + 金）=====
GREEN = '#1a5632'
GOLD = '#c8960c'
GREEN_LIGHT = '#2d8a4e'
GOLD_LIGHT = '#e8b84c'

# ====== 绘图 ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
fig.patch.set_facecolor('#faf8f5')

# ---- 左图：Loss 曲线 ----
ax1.plot(range(1, len(loss)+1), loss, color=GREEN, linewidth=1.5, zorder=3)
ax1.fill_between(range(1, len(loss)+1), loss, alpha=0.08, color=GREEN)
ax1.set_xlabel('迭代次数 (Iteration)', fontsize=10, color='#333333')
ax1.set_ylabel('MultiClass Log Loss', fontsize=10, color='#333333')
ax1.set_title('训练 Loss 曲线', fontsize=12, fontweight='bold', color=GREEN, pad=10)

# 标注关键节点
nodes = {50: 1.8682, 100: 1.6704, 200: 1.4406, 300: 1.2904, 500: 1.0707, 1000: 0.7476}
for it, val in nodes.items():
    ax1.scatter(it, val, color=GOLD, s=28, zorder=4, edgecolors='white', linewidth=0.6)
    ax1.annotate(f'{val:.3f}', xy=(it, val), xytext=(0, 10),
                 textcoords='offset points', ha='center', fontsize=7.5,
                 color='#555555', fontweight='normal')

# 初始和最终标注
ax1.annotate(f'初始: {loss[0]:.3f}', xy=(1, loss[0]), xytext=(-30, 12),
             textcoords='offset points', fontsize=7.5, color='#888888', ha='left')
ax1.annotate(f'最终: {loss[-1]:.3f}', xy=(1000, loss[-1]), xytext=(-60, -18),
             textcoords='offset points', fontsize=7.5, color=GREEN, ha='left', fontweight='bold')

ax1.set_xlim(0, 1050)
ax1.set_ylim(0, 3.2)
ax1.grid(True, alpha=0.25, linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#cccccc')
ax1.spines['bottom'].set_color('#cccccc')
ax1.tick_params(colors='#666666', labelsize=8)
ax1.xaxis.set_major_locator(MaxNLocator(6))

# ---- 右图：特征重要性 ----
bars = ax2.barh(feat_names_cn, feat_values, height=0.48, color=GREEN,
                edgecolor='white', linewidth=0.6, zorder=3)

# 最高特征用金色高亮
bars[-1].set_color(GOLD)
bars[-1].set_edgecolor('white')

# 在条上标注数值
for i, (bar, val) in enumerate(zip(bars, feat_values)):
    ax2.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=7.5, color='#555555')

ax2.set_xlabel('特征重要性 (Feature Importance)', fontsize=10, color='#333333')
ax2.set_title('Top-15 特征重要性', fontsize=12, fontweight='bold', color=GREEN, pad=10)
ax2.set_xlim(0, max(feat_values) * 1.25)
ax2.grid(True, axis='x', alpha=0.25, linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#cccccc')
ax2.spines['bottom'].set_color('#cccccc')
ax2.tick_params(colors='#666666', labelsize=8)

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=GREEN, label='其余特征'),
                   Patch(facecolor=GOLD, label='最重要特征')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=7.5,
           framealpha=0.8, edgecolor='#dddddd')

plt.tight_layout(pad=2.0)
plt.savefig('web_app/static/training_metrics.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
print(f'图已保存到 web_app/static/training_metrics.png')
print(f'尺寸: {fig.get_size_inches()}')
plt.close()