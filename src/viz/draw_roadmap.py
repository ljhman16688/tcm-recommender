# -*- coding: utf-8 -*-
"""绘制 KG-RAR 框架技术路线图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

# ===== 配色方案 =====
C_BG      = '#FAFBF8'    # 背景
C_DARK    = '#2D3A1F'    # 深绿（标题、边框）
C_GREEN1  = '#4A7C59'    # 主绿
C_GREEN2  = '#6B9E75'    # 浅绿
C_GREEN3  = '#8FBC94'    # 更浅
C_GOLD1   = '#C8A84E'    # 金色
C_GOLD2   = '#D4BC6A'    # 浅金
C_GOLD3   = '#E8D8A0'    # 淡金
C_WHITE   = '#FFFFFF'
C_GRAY    = '#888888'
C_LIGHT   = '#F0F3EB'    # 淡绿底
C_RED     = '#C0504D'    # 强调红
C_BLUE    = '#5B7BB5'    # 蓝色（LLM）

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(18, 24))
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(C_BG)

# ===== 辅助函数 =====
def draw_box(ax, x, y, w, h, text, color=C_GREEN1, text_color='white',
             fontsize=11, bold=True, edge_color=None, linewidth=2, alpha=1.0):
    """绘制圆角矩形"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.15", facecolor=color,
                         edgecolor=edge_color or C_DARK, linewidth=linewidth,
                         alpha=alpha)
    ax.add_patch(box)
    lines = text.split('\n')
    total_h = len(lines) * fontsize * 1.3
    y_start = y + total_h / 2 - fontsize * 0.65
    for i, line in enumerate(lines):
        ax.text(x, y_start - i * fontsize * 1.3, line, ha='center', va='top',
                fontsize=fontsize, color=text_color, fontweight='bold' if bold else 'normal')

def draw_arrow(ax, x1, y1, x2, y2, color=C_DARK, lw=2, style='->'):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle="arc3,rad=0"))

def draw_label(ax, x, y, text, fontsize=10, color=C_DARK, ha='center', bold=False):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=color,
            fontweight='bold' if bold else 'normal')

def draw_sub_box(ax, x, y, w, h, text, color=C_GREEN3, text_color=C_DARK, fontsize=9):
    """绘制子模块小框"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.08", facecolor=color,
                         edgecolor=C_GREEN1, linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    lines = text.split('\n')
    total_h = len(lines) * fontsize * 1.3
    y_start = y + total_h / 2 - fontsize * 0.65
    for i, line in enumerate(lines):
        ax.text(x, y_start - i * fontsize * 1.3, line, ha='center', va='top',
                fontsize=fontsize, color=text_color)

# ===== 系统边界大框 =====
boundary = FancyBboxPatch((0.3, 0.3), 17.4, 23.4,
                          boxstyle="round,pad=0.3", facecolor='none',
                          edgecolor=C_GRAY, linewidth=2, linestyle='--', alpha=0.5)
ax.add_patch(boundary)
draw_label(ax, 1.2, 23.2, 'KG-RAR 中药剂型智能推荐框架', fontsize=14, color=C_GRAY, bold=True)

# ===== 第1层：数据层 =====
Y1 = 22.3
draw_box(ax, 9, Y1, 15.5, 2.0, '', color=C_LIGHT, text_color=C_DARK, fontsize=10, edge_color=C_GRAY, alpha=0.7)
draw_label(ax, 9, Y1 + 0.75, '| 数据层：多源数据融合与知识图谱构建', fontsize=12, color=C_DARK, bold=True)

# 子模块
draw_sub_box(ax, 3.5, Y1 - 0.3, 5.0, 1.1,
             '中国药典成方制剂\n1,572 条处方-剂型配对',
             C_GOLD3, C_DARK, 9)
draw_sub_box(ax, 9, Y1 - 0.3, 5.0, 1.1,
             '部颁标准中药成方制剂\n3,484 条处方-剂型配对',
             C_GOLD3, C_DARK, 9)
draw_sub_box(ax, 14.5, Y1 - 0.3, 5.0, 1.1,
             '中药知识图谱\n16,031 实体 · 71,571 三元组',
             C_GREEN3, C_DARK, 9)

# 箭头
draw_arrow(ax, 6.1, Y1 - 0.3, 6.6, Y1 - 0.3, C_GRAY, 1.5, '->')
draw_arrow(ax, 11.6, Y1 - 0.3, 12.1, Y1 - 0.3, C_GRAY, 1.5, '->')

# ===== 第2层：特征工程 =====
Y2 = 18.5
draw_box(ax, 9, Y2, 15.5, 2.5, '', color=C_LIGHT, text_color=C_DARK, fontsize=10, edge_color=C_GRAY, alpha=0.7)
draw_label(ax, 9, Y2 + 1.1, '| 特征层：68维手工特征 + 41维KG特征 → 109维融合特征', fontsize=12, color=C_DARK, bold=True)

# 手工特征
draw_sub_box(ax, 4, Y2 - 0.1, 6.2, 1.5,
             '-- 68维手工特征\n  理化(8)+四气五味(12)+归经(12)+质地(6)\n  +证候(10)+疾病科室(8)+剂量(8)+急缓(4)',
             C_GREEN2, 'white', 8.5)
# KG特征
draw_sub_box(ax, 14, Y2 - 0.1, 6.2, 1.5,
             '-- 41维KG特征\n  药材历史剂型偏好(19)\n  +适应症-剂型关联(19)+聚合统计(3)',
             C_GOLD2, C_DARK, 8.5)

# 融合
draw_label(ax, 9, Y2 - 0.1, '⊕', fontsize=20, color=C_DARK, bold=True)

# 箭头：数据层 → 特征层
draw_arrow(ax, 9, Y1 - 1.3, 9, Y2 + 1.5, C_GRAY, 2.5, '->')

# ===== 第3层：CatBoost 分类器 =====
Y3 = 15.2
draw_box(ax, 9, Y3, 9.0, 1.5, '', color=C_GREEN1, edge_color=C_DARK, alpha=0.15)
draw_box(ax, 9, Y3, 8.8, 1.3, 'CatBoost 梯度提升多分类器\n有序提升 · 对称树 · 原生类别特征支持',
         C_GREEN1, 'white', 11)

# 箭头：特征层 → 分类器
draw_arrow(ax, 9, Y2 - 1.6, 9, Y3 + 0.95, C_DARK, 2.5, '->')
draw_label(ax, 10.2, (Y2 - 1.6 + Y3 + 0.95) / 2, 'xi in R^109', fontsize=9, color=C_GRAY, ha='left')

# ===== 第4层：KG锚定机制 =====
Y4 = 12.8
draw_box(ax, 9, Y4, 10.0, 1.2, '', color=C_GOLD1, edge_color=C_DARK, alpha=0.15)
draw_box(ax, 9, Y4, 9.8, 1.0, 'KG 锚定机制\n输入处方药材与 KG 已知处方完全一致 → 标准剂型强制置顶',
         C_GOLD1, C_DARK, 10)

# 箭头
draw_arrow(ax, 9, Y3 - 0.9, 9, Y4 + 0.8, C_DARK, 2.5, '->')
draw_label(ax, 10.2, (Y3 - 0.9 + Y4 + 0.8) / 2, 'Top-5\n候选', fontsize=9, color=C_GRAY, ha='left')

# ===== 第5层：KG-RAG推理流水线 =====
Y5 = 9.5
draw_box(ax, 9, Y5, 15.5, 2.5, '', color=C_LIGHT, text_color=C_DARK, fontsize=10, edge_color=C_GRAY, alpha=0.7)
draw_label(ax, 9, Y5 + 1.1, '| 推理层：KG-RAG 检索增强生成流水线', fontsize=12, color=C_DARK, bold=True)

# RAG三步
draw_sub_box(ax, 2.8, Y5 - 0.3, 4.8, 1.8,
             '① Retrieve  检索\n  药材属性 + 相似处方\n  + 疾病偏好 + 剂型规则',
             C_GREEN3, C_DARK, 9)
draw_sub_box(ax, 9, Y5 - 0.3, 4.8, 1.8,
             '② Augment  增强\n  分类器Top-5 + 四路证据\n  → 结构化提示词',
             C_GOLD3, C_DARK, 9)
draw_sub_box(ax, 15.2, Y5 - 0.3, 4.8, 1.8,
             '③ Generate  生成\n  大语言模型推理\n  → JSON格式推荐结果',
             C_BLUE, 'white', 9)

# RAG 之间的箭头
draw_arrow(ax, 5.3, Y5 - 0.3, 6.7, Y5 - 0.3, C_GRAY, 2, '->')
draw_arrow(ax, 11.5, Y5 - 0.3, 12.9, Y5 - 0.3, C_GRAY, 2, '->')

# 箭头：KG锚定 → RAG
draw_arrow(ax, 9, Y4 - 0.8, 9, Y5 + 1.5, C_DARK, 2.5, '->')

# KG → RAG 侧边箭头（知识检索回路）
ax.annotate('', xy=(16.8, Y5 - 0.3), xytext=(16.8, Y1 - 0.3),
            arrowprops=dict(arrowstyle='->', color=C_GOLD1, lw=2.5,
                           connectionstyle="arc3,rad=0.5"))
draw_label(ax, 17.3, (Y1 - 0.3 + Y5 - 0.3) / 2, '知识\n检索', fontsize=9, color=C_GOLD1, ha='left', bold=True)

# ===== 第6层：输出 =====
Y6 = 6.2
draw_box(ax, 9, Y6, 8.5, 1.2, '', color=C_DARK, edge_color=C_DARK, alpha=1.0)
draw_box(ax, 9, Y6, 8.3, 1.0,
         '端到端可解释剂型推荐\n排名列表 + 置信度 + 推理依据 + 排除原因',
         C_DARK, C_GOLD3, 11)

# 箭头
draw_arrow(ax, 9, Y5 - 1.5, 9, Y6 + 0.8, C_DARK, 2.5, '->')

# ===== 底部：LLM对比标注 =====
Y7 = 4.5
draw_box(ax, 9, Y7, 14.0, 0.8, '', color=C_RED, edge_color=C_RED, alpha=0.1)
draw_label(ax, 9, Y7,
           'Top-1 = 55.5%  vs  GPT-4o 32.7%  ·  KG特征以41维超越手工特征12.7pp  ·  Macro-F1 = 0.416',
           fontsize=10.5, color=C_RED, bold=True)

draw_arrow(ax, 9, Y6 - 0.8, 9, Y7 + 0.55, C_GRAY, 1.5, '->')

# ===== 图例 =====
LY = 2.5
draw_box(ax, 2.5, LY, 1.2, 0.6, '', C_GREEN1, 'white', 7)
draw_label(ax, 3.5, LY, 'CatBoost / 特征提取', fontsize=9, color=C_DARK, ha='left')

draw_box(ax, 7.5, LY, 1.2, 0.6, '', C_GOLD1, C_DARK, 7)
draw_label(ax, 8.5, LY, 'KG锚定 / 知识检索', fontsize=9, color=C_DARK, ha='left')

draw_box(ax, 12.0, LY, 1.2, 0.6, '', C_BLUE, 'white', 7)
draw_label(ax, 13.0, LY, '大语言模型 / 推理', fontsize=9, color=C_DARK, ha='left')

# ===== 标题 =====
ax.text(9, 24.3, '基于知识图谱检索增强的中药剂型推荐方法（KG-RAR）技术路线',
        ha='center', va='center', fontsize=17, color=C_DARK, fontweight='bold')

# ===== 保存 =====
plt.tight_layout(pad=0)
out_path = 'outputs/figures/fig0_technical_roadmap.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=C_BG, edgecolor='none')
plt.close()
print(f'技术路线图已保存到: {out_path}')