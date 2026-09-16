# -*- coding: utf-8 -*-
"""Dosage ablation: 109-dim (with dose) vs 101-dim (no dose) — 18 standard forms only"""
import sys, io, pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
import warnings; warnings.filterwarnings('ignore')

# ==============================================
# 1. Load & filter to 18 standard forms
# ==============================================
print('[1/3] Loading dataset...')
df = pd.read_csv('outputs/datasets/Dataset3_merged_features.csv')

# 18 standard forms (exclude 膜剂, 滴丸, 贴膏剂, 软胶囊)
STANDARD_18 = ['丸剂', '片剂', '胶囊剂', '颗粒剂', '口服液', '散剂', '膏剂', '糖浆剂',
               '注射剂', '酒剂', '酊剂', '栓剂', '茶剂', '丹剂', '搽剂', '洗剂',
               '气雾剂', '露剂']
df = df[df['form'].isin(STANDARD_18)].copy()
print('Samples: %d, Classes: %d' % (len(df), df['form'].nunique()))
print('Classes: %s' % df['form'].unique().tolist())

meta_cols = ['drug_name', 'herb_list', 'indication_text']
df = df.drop(columns=[c for c in meta_cols if c in df.columns])

le = LabelEncoder()
y = le.fit_transform(df['form'])
form_col = df.pop('form')

# ==============================================
# 2. Split (7:1:2)
# ==============================================
print('[2/3] Splitting...')
cc = pd.Series(y).value_counts()
ys = y.copy()
for rc in cc[cc < 3].index:
    ys[y == rc] = 999
X_temp, X_test, y_temp, y_test = tts(
    df, y, test_size=0.2, stratify=ys, random_state=42)
yti = y_temp.copy()
cc2 = pd.Series(yti).value_counts()
ys2 = yti.copy()
for rc in cc2[cc2 < 3].index:
    ys2[yti == rc] = 999
X_train, X_val, y_train, y_val = tts(
    X_temp, y_temp, test_size=0.125, stratify=ys2, random_state=42)
print('Train: %d | Val: %d | Test: %d' % (len(X_train), len(X_val), len(y_test)))

# ==============================================
# 3. Feature columns
# ==============================================
DOSE_COLS = ['total_weight', 'avg_dose', 'max_dose', 'min_dose',
             'dose_std', 'dose_range', 'max_min_ratio', 'log_total_weight']
all_cols = list(df.columns)
kg_cols = [c for c in all_cols if c.startswith('kg_')]
hc_cols = [c for c in all_cols if not c.startswith('kg_')]
hc_no_dose_cols = [c for c in hc_cols if c not in DOSE_COLS]

# 68-dim handcrafted only (no KG features)
cols_with_dose = hc_cols
cols_no_dose = hc_no_dose_cols

print('Handcrafted only: %d dims (with dose) | %d dims (no dose)' %
      (len(cols_with_dose), len(cols_no_dose)))

# ==============================================
# 4. Run CatBoost
# ==============================================
def run_cb(feature_cols, label):
    X = df[feature_cols].fillna(0).copy()
    if 'urgency' in X.columns:
        X['urgency'] = X['urgency'].astype(str)
    Xtr = X.loc[X_train.index]
    Xv = X.loc[X_val.index]
    Xte = X.loc[X_test.index]
    ci = [list(X.columns).index('urgency')] if 'urgency' in X.columns else []
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        random_seed=42, loss_function='MultiClass', eval_metric='MultiClass',
        cat_features=ci, early_stopping_rounds=50, verbose=0)
    model.fit(Pool(Xtr, y_train, cat_features=ci),
              eval_set=Pool(Xv, y_val, cat_features=ci), plot=False)
    yp = model.predict_proba(Xte)
    ypi = yp.argmax(axis=1)
    t1 = (ypi == y_test).mean()
    t3 = np.any(np.argsort(yp, 1)[:, -3:] == y_test[:, None], 1).mean()
    t5 = np.any(np.argsort(yp, 1)[:, -5:] == y_test[:, None], 1).mean()
    ranks = np.argsort(np.argsort(-yp, 1), 1)
    mr = (1.0 / (ranks[np.arange(len(y_test)), y_test] + 1)).mean()
    ypb = np.zeros_like(yp); ypb[np.arange(len(ypi)), ypi] = 1
    ytb = np.zeros_like(yp); ytb[np.arange(len(y_test)), y_test] = 1
    mic = f1_score(ytb, ypb, average='micro')
    mac = f1_score(ytb, ypb, average='macro')
    per_class_f1 = f1_score(y_test, ypi, average=None, labels=range(len(le.classes_)), zero_division=0)
    print('%-50s %4dd  T1=%.4f T3=%.4f T5=%.4f MRR=%.4f Mic=%.4f Mac=%.4f' %
          (label, len(feature_cols), t1, t3, t5, mr, mic, mac))
    return t1, t3, t5, mr, mic, mac, per_class_f1

print('[3/3] Running dosage ablation (68-dim vs 60-dim, handcrafted only)...')
print()
r_w = run_cb(cols_with_dose, 'CatBoost (68-dim, with dose)')
r_wo = run_cb(cols_no_dose, 'CatBoost (60-dim, no dose)')

# ==============================================
# 5. Results
# ==============================================
print()
print('=' * 90)
print('DOSAGE ABLATION RESULTS (18 standard forms, 68-dim vs 60-dim, handcrafted only)')
print('=' * 90)
hdr = '%-50s %5s  %8s %8s %8s %8s %10s %10s'
print(hdr % ('Model', 'Dim', 'Top-1', 'Top-3', 'Top-5', 'MRR', 'Micro-F1', 'Macro-F1'))
print('-' * 90)
for label, dim, r in [('CatBoost (101-dim, no dose)', len(cols_no_dose), r_wo[:6]),
                       ('CatBoost (109-dim, with dose)', len(cols_with_dose), r_w[:6])]:
    print('%-50s %5d  %8.4f %8.4f %8.4f %8.4f %10.4f %10.4f' % ((label, dim) + r))
delta = tuple(r_w[i] - r_wo[i] for i in range(6))
print('%-50s %5s  %+8.4f %+8.4f %+8.4f %+8.4f %+10.4f %+10.4f' % (('Δ (dose contribution)', '',) + delta))

print()
print('=' * 90)
print('PER-CLASS F1 COMPARISON (18 standard forms)')
print('=' * 90)
print('%-20s %8s %8s %8s %8s' % ('Form', 'Samples', 'F1(101d)', 'F1(109d)', 'Δ'))
print('-' * 60)
class_names = le.classes_
test_counts = pd.Series(y_test).value_counts().sort_index()
per_class_w = r_w[6]
per_class_wo = r_wo[6]
for ci in range(len(class_names)):
    cnt = test_counts.get(ci, 0)
    f1_w = per_class_w[ci]
    f1_wo = per_class_wo[ci]
    d = f1_w - f1_wo
    marker = ' <<' if abs(d) > 0.05 else ''
    print('%-20s %8d %8.4f %8.4f %+8.4f%s' % (class_names[ci], cnt, f1_wo, f1_w, d, marker))

print()
print('DONE')