# -*- coding: utf-8 -*-
"""冻结 712 部署资源并端到端验证（2026-10-14，部署重构第一步）。

产出：
  1. data/deploy/pmic_agg_712.json —— 仅训练集(3573)的 PMI 聚合统计（JSON 序列化），
     部署端无需在运行时依赖 Excel 与协议重建管线；
  2. web_app/catboost_G.cbm —— 主模型 G（从 outputs/final_model_G/ 复制，随镜像打包）。

验证（关键）：用「JSON 反序列化回来的聚合 + G 模型」重算 1021 条测试集，
Top-1 必须复现论文口径 0.4515（±1e-4），证明部署端特征链与训练端逐位一致。
再跑一个前端示例处方的 Top-5 冒烟。
"""
import sys, io, os, json, shutil
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from catboost import CatBoostClassifier
from src.data.protocol_712 import build_records, split_records
from src.features.extract_features import ALL_FORMS, parse_prescription
from src.features.kg_pmi_features import build_aggregates_pmi
from src.deploy_712 import save_agg, load_agg, build_row

AGG_PATH = os.path.join(_ROOT, 'data', 'deploy', 'pmic_agg_712.json')
MODEL_SRC = os.path.join(_ROOT, 'outputs', 'final_model_G', 'catboost_G.cbm')
MODEL_DST = os.path.join(_ROOT, 'web_app', 'catboost_G.cbm')

# ---- 1. 重建 712 数据并断言规模 ----
records = build_records(verbose=False)
split = split_records(records, verbose=False)
train_recs, val_recs, test_recs = split['train'], split['val'], split['test']
assert len(records) == 5105 and len(train_recs) == 3573 and len(val_recs) == 511 and len(test_recs) == 1021, \
    '712 划分规模不符'
print('712 划分: 训练 %d / 验证 %d / 测试 %d（%d 类）' % (
    len(train_recs), len(val_recs), len(test_recs), len(split['le'].classes_)))

# ---- 2. 冻结仅训练聚合 ----
agg = build_aggregates_pmi(train_recs, ALL_FORMS)
os.makedirs(os.path.dirname(AGG_PATH), exist_ok=True)
save_agg(agg, AGG_PATH, len(train_recs))
agg_loaded = load_agg(AGG_PATH)  # 反序列化回读（验证 JSON 往返无损）

# ---- 3. 加载 G 模型，用回读聚合重算测试集 ----
model = CatBoostClassifier()
model.load_model(MODEL_SRC)
# cbm 内存储整数类号 0..18（训练时 y=LabelEncoder 编码，le.fit(sorted(set(forms)))），
# 类名顺序 = sorted(ALL_FORMS)，与 model.classes_ 一一对应。
classes = sorted(ALL_FORMS)
feat_cols = list(model.feature_names_)
assert len(feat_cols) == 109, '特征列应为 109 维，实际 %d' % len(feat_cols)
assert len(model.classes_) == 19, '模型类别应为 19 类'
with open(os.path.join(_ROOT, 'outputs', 'final_model_G', 'test_metrics.json'), encoding='utf-8') as f:
    _tm = json.load(f)
assert list(_tm['classes']) == classes, 'test_metrics.classes 与 sorted(ALL_FORMS) 不一致'
print('G 模型: %d 维 / %d 类（类名顺序 = sorted(ALL_FORMS)）' % (len(feat_cols), len(classes)))

Xte = pd.DataFrame([build_row(r['herbs'], r['doses'], r['indication'], r['admin'], agg_loaded)
                    for r in test_recs])
Xte = Xte.reindex(columns=feat_cols).fillna(0)
probs = model.predict_proba(Xte)
y_test = split['y'][split['test_idx']]
top1 = float((probs.argmax(axis=1) == y_test).mean())
ok = abs(top1 - 0.4515) < 1e-4
print('冻结聚合重算测试集 Top-1 = %.4f（期望 0.4515）→ %s' % (top1, 'PASS' if ok else 'FAIL'))

# ---- 4. 前端示例冒烟（与 /api/predict 入口一致） ----
example = '丹参 15g, 三七 9g, 冰片 0.5g'
herbs, doses = parse_prescription(example)
row = build_row(herbs, doses, '冠心病心绞痛，气滞血瘀证', '', agg_loaded)
frame = pd.DataFrame([row]).reindex(columns=feat_cols).fillna(0)
p = model.predict_proba(frame)[0]
top = sorted(zip(classes, p.tolist()), key=lambda x: -x[1])[:5]
print('示例「%s」Top-5: %s' % (example, [(c, round(float(v) * 100, 1)) for c, v in top]))

# ---- 5. 复制模型到 web_app（随镜像打包） ----
shutil.copyfile(MODEL_SRC, MODEL_DST)
print('已复制模型 -> %s' % os.path.relpath(MODEL_DST, _ROOT))

if not ok:
    print('FAIL: 冻结聚合重算未复现 45.15%，部署不得上线，请排查后再运行本脚本')
    sys.exit(1)
print('DONE: 部署资源冻结完成且端到端验证通过')