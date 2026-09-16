# -*- coding: utf-8 -*-
"""
生成论文全部图表：fig1 消融实验, fig2 LLM 对比, fig3 剂量消融
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

GREEN = '#1a5632'
GREEN_MID = '#2d8a4e'
GREEN_LIGHT = '#5ba872'
GREEN_PALE = '#a8d5ba'
GOLD = '#c8960c'
GOLD_LIGHT = '#e8b84c'
GRAY = '#b0b0b0'
GRAY_LIGHT = '#d5d5d5'

# ================================================================
# fig1: 消融实验结果
# ================================================================
models_ab = ['CatBoost\n(手工68维)', 'CatBoost\n(KG-only 41维)', 'CatBoost\n(手工+KG 109维)', 'BERT-only\n(768维)']
top1_ab = [42.5, 52.5, 55.5, 38.9]
macro_f1_ab = [25.4, 39.4, 41.6, 10.9]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
fig.patch.set_facecolor('#ffffff')

x = np.arange(len(models_ab))
width = 0.55

colors_ab = [GREEN_MID, GREEN_LIGHT, GOLD, GRAY_LIGHT]
bars1 = ax1.bar(x, top1_ab, width, color=colors_ab, edgecolor='white', linewidth=0.8, zorder=3)
for bar, val in zip(bars1, top1_ab):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#333333')

ax1.set_ylabel('Top-1 准确率 (%)', fontsize=11, color='#333333')
ax1.set_title('(a) Top-1 准确率对比', fontsize=12, fontweight='bold', color='#222222', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(models_ab, fontsize=8.5, color='#333333')
ax1.set_ylim(0, 72)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(15))
ax1.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#cccccc')
ax1.spines['bottom'].set_color('#cccccc')
ax1.tick_params(colors='#666666', labelsize=8.5)

bars2 = ax2.bar(x, macro_f1_ab, width, color=colors_ab, edgecolor='white', linewidth=0.8, zorder=3)
for bar, val in zip(bars2, macro_f1_ab):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#333333')

ax2.set_ylabel('Macro-F1 (%)', fontsize=11, color='#333333')
ax2.set_title('(b) Macro-F1 对比', fontsize=12, fontweight='bold', color='#222222', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(models_ab, fontsize=8.5, color='#333333')
ax2.set_ylim(0, 56)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(12))
ax2.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#cccccc')
ax2.spines['bottom'].set_color('#cccccc')
ax2.tick_params(colors='#666666', labelsize=8.5)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=GREEN_MID, label='手工特征'),
    Patch(facecolor=GREEN_LIGHT, label='KG 特征'),
    Patch(facecolor=GOLD, label='融合特征 (本文)'),
    Patch(facecolor=GRAY_LIGHT, label='端到端语义'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=4,
           fontsize=8, frameon=True, framealpha=0.9, edgecolor='#dddddd',
           bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.93])
plt.savefig('outputs/figures/fig1_ablation.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('fig1_ablation.png saved')
plt.close()

# ================================================================
# fig2: LLM 零样本对比
# ================================================================
# Load benchmark results
bench_path = 'data/llm_benchmark_v2.json'
if os.path.exists(bench_path):
    with open(bench_path, 'r', encoding='utf-8') as f:
        bench = json.load(f)
else:
    # Fallback: use hardcoded data from paper Table 2
    bench = {
        'Gemini-3.1': {'top1': 0.4510, 'top3': 0.7745, 'top5': 0.9130, 'ndcg': 0.6933, 'mrr': 0.6204, 'per_class_recall': {}},
        'Qwen-3.7': {'top1': 0.4154, 'top3': 0.7250, 'top5': 0.8843, 'ndcg': 0.6596, 'mrr': 0.5852, 'per_class_recall': {}},
        'GLM-5.1': {'top1': 0.4085, 'top3': 0.7448, 'top5': 0.8961, 'ndcg': 0.6660, 'mrr': 0.5895, 'per_class_recall': {}},
        'MiniMax-M3': {'top1': 0.3947, 'top3': 0.7280, 'top5': 0.8863, 'ndcg': 0.6521, 'mrr': 0.5744, 'per_class_recall': {}},
        'DeepSeek-V4': {'top1': 0.3729, 'top3': 0.6301, 'top5': 0.7527, 'ndcg': 0.5705, 'mrr': 0.5103, 'per_class_recall': {}},
        'GPT-4o': {'top1': 0.3225, 'top3': 0.5826, 'top5': 0.7676, 'ndcg': 0.5454, 'mrr': 0.4727, 'per_class_recall': {}},
    }

# CatBoost results
cb = {'top1': 0.5553, 'top3': 0.8495, 'top5': 0.9272, 'ndcg': 0.7131, 'mrr': 0.7131, 'per_class_recall': {}}

llm_names = list(bench.keys())
llm_top1 = [bench[n]['top1'] * 100 for n in llm_names]
llm_top5 = [bench[n]['top5'] * 100 for n in llm_names]
llm_ndcg = [bench[n]['ndcg'] * 100 for n in llm_names]
llm_mrr = [bench[n]['mrr'] * 100 for n in llm_names]

# Add CatBoost
all_names = llm_names + ['CatBoost\n(本文)']
all_top1 = llm_top1 + [cb['top1'] * 100]
all_top5 = llm_top5 + [cb['top5'] * 100]
all_ndcg = llm_ndcg + [cb['ndcg'] * 100]
all_mrr = llm_mrr + [cb['mrr'] * 100]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
fig.patch.set_facecolor('#ffffff')

x = np.arange(len(all_names))
width = 0.35

# Left: Top-1 + Top-5 grouped bars
colors_llm = [GRAY_LIGHT] * 6 + [GOLD]
bars_t1 = ax1.bar(x - width/2, all_top1, width, label='Top-1', color=GREEN_MID, edgecolor='white', linewidth=0.6, zorder=3)
bars_t5 = ax1.bar(x + width/2, all_top5, width, label='Top-5', color=GREEN_PALE, edgecolor='white', linewidth=0.6, zorder=3)

for bar, val in zip(bars_t1, all_top1):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='#333333')
for bar, val in zip(bars_t5, all_top5):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='#666666')

# Highlight CatBoost
ax1.axvline(x=5.5, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

ax1.set_ylabel('命中率 (%)', fontsize=11, color='#333333')
ax1.set_title('(a) Top-1 / Top-5 命中率对比', fontsize=12, fontweight='bold', color='#222222', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(all_names, fontsize=8, color='#333333')
ax1.set_ylim(0, 108)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(25))
ax1.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#cccccc')
ax1.spines['bottom'].set_color('#cccccc')
ax1.tick_params(colors='#666666', labelsize=8)
ax1.legend(loc='upper left', fontsize=8, framealpha=0.9, edgecolor='#dddddd')

# Right: NDCG + MRR grouped bars
bars_ndcg = ax2.bar(x - width/2, all_ndcg, width, label='NDCG@5', color=GREEN_MID, edgecolor='white', linewidth=0.6, zorder=3)
bars_mrr = ax2.bar(x + width/2, all_mrr, width, label='MRR', color=GREEN_PALE, edgecolor='white', linewidth=0.6, zorder=3)

for bar, val in zip(bars_ndcg, all_ndcg):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='#333333')
for bar, val in zip(bars_mrr, all_mrr):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='#666666')

ax2.axvline(x=5.5, color='#888888', linestyle='--', linewidth=0.8, alpha=0.5)

ax2.set_ylabel('排序质量 (%)', fontsize=11, color='#333333')
ax2.set_title('(b) NDCG@5 / MRR 对比', fontsize=12, fontweight='bold', color='#222222', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(all_names, fontsize=8, color='#333333')
ax2.set_ylim(0, 108)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(25))
ax2.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#cccccc')
ax2.spines['bottom'].set_color('#cccccc')
ax2.tick_params(colors='#666666', labelsize=8)
ax2.legend(loc='upper left', fontsize=8, framealpha=0.9, edgecolor='#dddddd')

plt.tight_layout(pad=2.0)
plt.savefig('outputs/figures/fig2_llm_compare.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('fig2_llm_compare.png saved')
plt.close()

# ================================================================
# fig3: 剂量特征消融
# ================================================================
forms = ['散剂', '胶囊剂', '丸剂', '片剂', '颗粒剂', '口服液', '膏剂', '糖浆剂', '丹剂']
f1_without = [13.3, 19.4, 32.0, 28.0, 25.0, 10.0, 18.0, 15.0, 0.0]
f1_with = [25.0, 7.9, 31.0, 27.0, 24.0, 11.0, 17.0, 14.0, 0.0]

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=200)
fig.patch.set_facecolor('#ffffff')

x = np.arange(len(forms))
width = 0.35

bars_wo = ax.bar(x - width/2, f1_without, width, label='不含剂量特征', color=GRAY_LIGHT,
                 edgecolor='white', linewidth=0.6, zorder=3)
bars_w = ax.bar(x + width/2, f1_with, width, label='含剂量特征', color=GREEN_MID,
                edgecolor='white', linewidth=0.6, zorder=3)

# Annotate changes
for i, (wo, w) in enumerate(zip(f1_without, f1_with)):
    diff = w - wo
    if abs(diff) >= 5:
        color = GREEN if diff > 0 else '#c0392b'
        arrow = '↑' if diff > 0 else '↓'
        ax.annotate(f'{arrow}{abs(diff):.1f}pp',
                    xy=(x[i] + width/2, w), xytext=(x[i] + width/2 + 0.1, w + 2),
                    fontsize=8, color=color, fontweight='bold', ha='center')

for bar, val in zip(bars_wo, f1_without):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='#888888')
for bar, val in zip(bars_w, f1_with):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='#333333')

ax.set_ylabel('F1 (%)', fontsize=11, color='#333333')
ax.set_title('剂量特征消融：各类剂型 F1 变化', fontsize=12, fontweight='bold', color='#222222', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(forms, fontsize=9, color='#333333')
ax.set_ylim(0, 42)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid(True, axis='y', alpha=0.25, linewidth=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.tick_params(colors='#666666', labelsize=8.5)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#dddddd')

plt.tight_layout(pad=2.0)
plt.savefig('outputs/figures/fig3_dosage.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('fig3_dosage.png saved')
plt.close()

print('\n所有图表生成完毕！')