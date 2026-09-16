# -*- coding: utf-8 -*-
"""Knowledge graph visualization - herb-centric subgraph"""
import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

with open('data/tcm_knowledge_graph.json', 'r', encoding='utf-8') as f:
    kg = json.load(f)

# ===== Pick a focal herb =====
focal_herb = '丹参'

# Find focal herb entity ID
herb_id = None
for eid, info in kg['entities'].items():
    if info.get('name') == focal_herb and info.get('type') == 'herb':
        herb_id = int(eid)
        break

if herb_id is None:
    print('Herb not found')
    exit()

# ===== Collect connected entities =====
neighbors = set()
neighbor_triples = []
for t in kg['triples']:
    h, r, tail = t['head'], t['relation'], t['tail']
    if h == herb_id or tail == herb_id:
        neighbors.add(h)
        neighbors.add(tail)
        neighbor_triples.append(t)

# For each neighbor, also get their dosage forms (if formula) or other connections
formula_neighbors = set()
formula_triples = []
for t in kg['triples']:
    h, r, tail = t['head'], t['relation'], t['tail']
    if h in neighbors and kg['entities'][str(tail)].get('type') == 'dosage_form':
        formula_triples.append(t)
        formula_neighbors.add(tail)

all_nodes = neighbors | formula_neighbors

# ===== Build positions (radial layout) =====
# Center: focal herb
# Layer 1: taste, meridian, qi, function
# Layer 2: formulas
# Layer 3: dosage forms

# Categorize nodes
node_info = {}
for nid in all_nodes:
    info = kg['entities'][str(nid)]
    node_info[nid] = info

# Pick a subset for cleaner visualization
# Max 20 formulas, top dosage forms
formula_ids = [f for f in neighbors if node_info[f]['type'] == 'formula'][:20]

# Build graph
nodes_to_draw = {herb_id}
for f in formula_ids:
    nodes_to_draw.add(f)
for t in neighbor_triples:
    h, r, tail = t['head'], t['relation'], t['tail']
    if h == herb_id and node_info[tail]['type'] in ['taste', 'meridian', 'qi']:
        nodes_to_draw.add(tail)

# Also add formula dosage forms
for t in kg['triples']:
    if t['head'] in formula_ids[:10] and node_info[t['tail']]['type'] == 'dosage_form':
        nodes_to_draw.add(t['tail'])

nodes_to_draw = list(nodes_to_draw)[:35]

# ===== Layout =====
n = len(nodes_to_draw)
pos = {}
node_labels = {}
node_colors = []
node_sizes = []

# Center node
center = nodes_to_draw.index(herb_id) if herb_id in nodes_to_draw else 0
pos[herb_id] = (0, 0)
node_labels[herb_id] = node_info[herb_id]['name']

# Layer 1: properties (taste, meridian, qi) - circle radius 1.5
prop_nodes = [n for n in nodes_to_draw if n != herb_id and node_info[n]['type'] in ['taste', 'meridian', 'qi']]
form_nodes = [n for n in nodes_to_draw if n != herb_id and node_info[n]['type'] == 'formula']
form2_nodes = [n for n in nodes_to_draw if n != herb_id and node_info[n]['type'] == 'dosage_form']

for i, nid in enumerate(prop_nodes):
    angle = 2 * np.pi * i / max(len(prop_nodes), 1)
    pos[nid] = (1.8 * np.cos(angle), 1.8 * np.sin(angle))
    node_labels[nid] = node_info[nid]['name']

for i, nid in enumerate(form_nodes):
    angle = 2 * np.pi * i / max(len(form_nodes), 1)
    pos[nid] = (4.0 * np.cos(angle), 4.0 * np.sin(angle))
    label = node_info[nid]['name']
    if len(label) > 8: label = label[:7] + '..'
    node_labels[nid] = label

for i, nid in enumerate(form2_nodes):
    angle = 2 * np.pi * i / max(len(form2_nodes), 1) + np.pi / len(form2_nodes)
    pos[nid] = (6.0 * np.cos(angle), 6.0 * np.sin(angle))
    node_labels[nid] = node_info[nid]['name']

# ===== Color mapping =====
type_colors = {
    'herb': '#E65100',
    'taste': '#2E7D32',
    'meridian': '#1565C0',
    'qi': '#6A1B9A',
    'formula': '#F57C00',
    'dosage_form': '#C62828',
    'function': '#00838F',
}

fig, ax = plt.subplots(1, 1, figsize=(14, 12))
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-7.5, 7.5)
ax.axis('off')

# Draw edges
for t in kg['triples']:
    h, r, tail = t['head'], t['relation'], t['tail']
    if h in nodes_to_draw and tail in nodes_to_draw:
        rel_name = t.get('relation_name', '')
        alpha = 0.3
        lw = 0.8
        color = '#999999'
        if rel_name == 'has_form':
            alpha = 0.6; lw = 1.2; color = '#C62828'
        elif rel_name == 'contained_in':
            alpha = 0.4; lw = 0.8; color = '#F57C00'
        ax.plot([pos[h][0], pos[tail][0]], [pos[h][1], pos[tail][1]],
                color=color, alpha=alpha, linewidth=lw, zorder=1)

# Draw nodes
for nid in nodes_to_draw:
    x, y = pos[nid]
    ntype = node_info[nid]['type']
    color = type_colors.get(ntype, '#999999')
    size = 300 if ntype == 'herb' else (180 if ntype == 'formula' else 120)

    ax.scatter(x, y, s=size, c=color, edgecolors='white', linewidth=1.2, zorder=2)
    offset_y = -0.25 if ntype in ['taste','meridian','qi'] else 0.3
    fs = 10 if ntype == 'herb' else (8 if ntype in ['formula','dosage_form'] else 7)
    ax.text(x, y + offset_y, node_labels[nid], ha='center', va='center',
            fontsize=fs, weight='bold' if ntype == 'herb' else 'normal',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# Legend
legend_elements = [
    mpatches.Patch(color='#E65100', label='Herb (药材)'),
    mpatches.Patch(color='#2E7D32', label='Taste (味)'),
    mpatches.Patch(color='#1565C0', label='Meridian (归经)'),
    mpatches.Patch(color='#6A1B9A', label='Qi (气)'),
    mpatches.Patch(color='#F57C00', label='Formula (处方)'),
    mpatches.Patch(color='#C62828', label='Dosage Form (剂型)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

ax.set_title('KG Subgraph: %s' % focal_herb, fontsize=16, weight='bold')

plt.tight_layout()
out = 'KG_visualization_%s.png' % focal_herb
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved:', out)
plt.close()

# ===== Also make an overview stats chart =====
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Entity types
etypes = kg['statistics']['entity_types']
sorted_types = sorted(etypes.items(), key=lambda x: -x[1])
labels = [t for t, _ in sorted_types]
values = [v for _, v in sorted_types]
colors_bar = ['#E65100','#1565C0','#2E7D32','#F57C00','#6A1B9A','#00838F','#C62828','#D84315']

bars = ax1.barh(range(len(labels)), values, color=colors_bar[:len(labels)])
ax1.set_yticks(range(len(labels)))
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel('Count', fontsize=11)
ax1.set_title('Entity Types', fontsize=13, weight='bold')
for bar, val in zip(bars, values):
    ax1.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9)

# Relation types
rtypes = kg['statistics']['relation_types']
sorted_rels = sorted(rtypes.items(), key=lambda x: -x[1])
ax2.pie([v for _, v in sorted_rels], labels=[k for k, _ in sorted_rels],
        autopct='%1.0f%%', colors=plt.cm.Set2.colors, startangle=90)
ax2.set_title('Relation Distribution', fontsize=13, weight='bold')

plt.tight_layout()
out2 = 'KG_overview.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
print('Saved:', out2)
plt.close()
