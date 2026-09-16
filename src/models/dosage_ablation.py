# -*- coding: utf-8 -*-
"""
剂量特征消融实验：有无剂量特征的 CatBoost 对比
数据源：药典数据.xlsx (1572条)
"""

import sys, io, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 1. 加载数据 + 解析药材和剂量
# ==============================================
def load_and_parse():
    df = pd.read_excel('data/药典数据.xlsx')
    print(f'Loaded: {len(df)} rows')

    # 提取剂型
    form_counts = df['dosage_form'].value_counts()
    print(f'Form types: {len(form_counts)}')

    # 解析处方文本：药材+剂量
    def parse_prescription(text):
        if pd.isna(text):
            return [], []
        text = str(text)
        # 匹配 "药材名 数字 g" 模式
        # 药材名可能含括号: 苍术（炒）, 厚朴(姜制)
        pattern = r'([一-龥（(][一-龥)）、(）]{0,8}?)\s*(\d+\.?\d*)\s*g'
        matches = re.findall(pattern, text)
        herbs = []
        doses = []
        for m in matches:
            name = m[0].strip()
            dose = float(m[1])
            # 清理药材名：去括号及内容
            name_clean = re.sub(r'[（(][^)）]*[)）]', '', name).strip()
            if len(name_clean) >= 2 and dose > 0:
                herbs.append(name_clean)
                doses.append(dose)
        return herbs, doses

    df[['herb_list', 'dose_list']] = df['prescription_raw'].apply(
        lambda x: pd.Series(parse_prescription(x))
    )

    # 剔除解析不出药材的行
    n_before = len(df)
    df = df[df['herb_list'].apply(len) > 0].copy()
    print(f'Removed {n_before - len(df)} rows with no herbs')
    print(f'Final: {len(df)} rows')

    return df

# ==============================================
# 2. 加载药材属性字典 + 提取特征
# ==============================================
def load_herb_dict():
    with open('data/herb_property_dict.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_features(df, herb_dict, include_dosage=False):
    """
    提取特征
    include_dosage: 是否包含剂量相关特征
    """
    features_list = []

    for _, row in df.iterrows():
        herbs = row['herb_list']
        doses = row['dose_list']
        n = len(herbs)
        if n == 0:
            continue

        feats = {}

        # === 药材数量 ===
        feats['num_herbs'] = n

        # === 药材属性聚合 ===
        nature_counter = {'寒':0,'凉':0,'平':0,'温':0,'热':0}
        taste_counter = {'辛':0,'甘':0,'酸':0,'苦':0,'咸':0,'淡':0,'涩':0}
        meridian_counter = {'心':0,'肝':0,'脾':0,'肺':0,'肾':0,'胃':0,'胆':0,'小肠':0,'大肠':0,'膀胱':0,'三焦':0,'心包':0}
        texture_counter = {'hard':0,'starchy':0,'fibrous':0,'mucilaginous':0,'oily':0,'gelatinous':0,'normal':0,'extract':0}
        count_volatile = 0
        count_toxic = 0
        count_mineral = 0
        count_precious = 0
        known = 0

        def normalize_herb(name):
            name = str(name).strip()
            name = re.sub(r'[（(][^)）]*[)）]', '', name)
            name = re.sub(r'^(?:生|鲜|炒|酒|醋|盐|姜|蜜|焦|煅|制|熟)', '', name)
            alias = {'萸肉':'山茱萸','法半夏':'半夏','清半夏':'半夏','姜半夏':'半夏',
                     '胆南星':'天南星','炙甘草':'甘草','炙黄芪':'黄芪',
                     '熟地黄':'地黄','生地黄':'地黄','醋柴胡':'柴胡'}
            return alias.get(name, name)

        for herb in herbs:
            herb_clean = normalize_herb(herb)
            props = herb_dict.get(herb, herb_dict.get(herb_clean))
            if props is None:
                texture_counter['normal'] += 1
                continue
            known += 1

            if props.get('volatile'): count_volatile += 1
            if props.get('toxic'): count_toxic += 1
            if props.get('mineral'): count_mineral += 1
            if props.get('precious'): count_precious += 1

            nat = props.get('nature','')
            if nat in nature_counter:
                nature_counter[nat] += 1
            for t in props.get('taste',[]):
                if t in taste_counter: taste_counter[t] += 1
            for m in props.get('meridian',[]):
                if m in meridian_counter: meridian_counter[m] += 1
            tex = props.get('texture','normal')
            if tex in texture_counter: texture_counter[tex] += 1

        # === 二值特征 ===
        feats['has_volatile'] = int(count_volatile > 0)
        feats['has_toxic'] = int(count_toxic > 0)
        feats['has_mineral'] = int(count_mineral > 0)
        feats['has_precious'] = int(count_precious > 0)

        # === 比例特征 ===
        feats['volatile_ratio'] = count_volatile / n
        feats['toxic_ratio'] = count_toxic / n
        feats['mineral_ratio'] = count_mineral / n
        feats['precious_ratio'] = count_precious / n
        feats['known_ratio'] = known / n

        # === 药性 ===
        for nat in ['寒','凉','平','温','热']:
            feats[f'nature_{nat}_ratio'] = nature_counter[nat] / n

        # === 药味 ===
        for t in ['辛','甘','酸','苦','咸','淡','涩']:
            feats[f'taste_{t}_ratio'] = taste_counter[t] / n

        # === 归经 ===
        for m in ['心','肝','脾','肺','肾','胃','胆','小肠','大肠','膀胱','三焦','心包']:
            feats[f'meridian_{m}_ratio'] = meridian_counter[m] / n

        # === 质地 ===
        for tex in ['hard','starchy','fibrous','mucilaginous','oily','gelatinous']:
            feats[f'texture_{tex}_ratio'] = texture_counter[tex] / n

        # === 适应症特征 ===
        ind_text = str(row['indication']) if pd.notna(row['indication']) else ''
        # urgency
        if any(w in ind_text for w in ['急性','急症','暴发','骤发']):
            feats['urgency'] = 'acute'
        elif any(w in ind_text for w in ['慢性','久','迁延','日久']):
            feats['urgency'] = 'chronic'
        else:
            feats['urgency'] = 'unspecified'
        # topical
        feats['is_topical'] = int(any(w in ind_text for w in ['外用','涂','敷','贴']))

        # === 剂量特征（仅在 Version B 中加入）===
        if include_dosage and len(doses) > 0:
            doses_arr = np.array(doses)
            feats['total_weight'] = doses_arr.sum()
            feats['avg_dose'] = doses_arr.mean()
            feats['max_dose'] = doses_arr.max()
            feats['min_dose'] = doses_arr.min()
            feats['dose_std'] = doses_arr.std()
            feats['dose_range'] = feats['max_dose'] - feats['min_dose']
            feats['max_min_ratio'] = feats['max_dose'] / feats['min_dose'] if feats['min_dose'] > 0 else 0
            # 总重对数（压缩极端值）
            feats['log_total_weight'] = np.log1p(feats['total_weight'])

        features_list.append(feats)

    return pd.DataFrame(features_list)

# ==============================================
# 3. 训练和评估
# ==============================================
def train_and_eval(X, y_labels, cat_features_idx):
    """跑一次 CatBoost 并返回指标"""
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([[l] for l in y_labels])

    # 处理极小类（合并用于分层采样）
    y_idx = y.argmax(axis=1)
    class_counts = pd.Series(y_idx).value_counts()
    rare = class_counts[class_counts < 3].index
    y_idx_strat = y_idx.copy()
    for rc in rare:
        y_idx_strat[y_idx == rc] = 999

    # 7:1:2 划分
    X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, stratify=y_idx_strat, random_state=42)

    y_idx_tv = y_tv.argmax(axis=1)
    cc = pd.Series(y_idx_tv).value_counts()
    rare2 = cc[cc < 3].index
    y_idx_tv_strat = y_idx_tv.copy()
    for rc in rare2:
        y_idx_tv_strat[y_idx_tv == rc] = 999

    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.125, stratify=y_idx_tv_strat, random_state=42)

    # 类别特征（仅 urgency）
    train_pool = Pool(X_train, y_train.argmax(axis=1), cat_features=cat_features_idx)
    val_pool = Pool(X_val, y_val.argmax(axis=1), cat_features=cat_features_idx)

    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        random_seed=42, loss_function='MultiClass', eval_metric='MultiClass',
        cat_features=cat_features_idx, early_stopping_rounds=50, verbose=0,
    )
    model.fit(train_pool, eval_set=val_pool, plot=False)

    # 评估
    y_test_idx = y_test.argmax(axis=1)
    y_proba = model.predict_proba(X_test)
    y_pred_idx = y_proba.argmax(axis=1)

    # Top-K
    top1 = (y_pred_idx == y_test_idx).mean()
    top3 = np.any(np.argsort(y_proba,axis=1)[:,-3:] == y_test_idx[:,None], axis=1).mean()
    top5 = np.any(np.argsort(y_proba,axis=1)[:,-5:] == y_test_idx[:,None], axis=1).mean()

    # MRR
    ranks = np.argsort(np.argsort(-y_proba,axis=1),axis=1)
    mrr = (1.0/(ranks[np.arange(len(y_test_idx)), y_test_idx]+1)).mean()

    # F1
    y_pred_bin = np.zeros_like(y_test)
    y_pred_bin[np.arange(len(y_pred_idx)), y_pred_idx] = 1
    micro_f1 = f1_score(y_test, y_pred_bin, average='micro')
    macro_f1 = f1_score(y_test, y_pred_bin, average='macro')

    # Per-class F1
    all_labels = np.union1d(np.unique(y_test_idx), np.unique(y_pred_idx))
    names = [mlb.classes_[i] for i in all_labels]
    report = classification_report(y_test_idx, y_pred_idx, labels=all_labels,
                                   target_names=names, output_dict=True, zero_division=0)
    per_class = {}
    for cls_name in names:
        if cls_name in report:
            per_class[cls_name] = round(report[cls_name]['f1-score'], 4)

    return {
        'top1': round(top1,4), 'top3': round(top3,4), 'top5': round(top5,4),
        'mrr': round(mrr,4), 'micro_f1': round(micro_f1,4), 'macro_f1': round(macro_f1,4),
        'per_class': per_class
    }, model

# ==============================================
# 4. 主流程
# ==============================================
if __name__ == '__main__':
    print('=' * 60)
    print('Dosage Feature Ablation Experiment')
    print('=' * 60)

    df = load_and_parse()
    herb_dict = load_herb_dict()
    y_labels = df['dosage_form'].tolist()

    # === Version A: 无剂量特征 ===
    print('\n' + '=' * 60)
    print('VERSION A: Without dosage features')
    print('=' * 60)
    X_a = extract_features(df, herb_dict, include_dosage=False)
    cat_idx_a = [list(X_a.columns).index('urgency')]
    results_a, model_a = train_and_eval(X_a, y_labels, cat_idx_a)

    print(f'Top-1: {results_a["top1"]:.4f}  Top-3: {results_a["top3"]:.4f}  '
          f'Top-5: {results_a["top5"]:.4f}  MRR: {results_a["mrr"]:.4f}')
    print(f'Micro-F1: {results_a["micro_f1"]:.4f}  Macro-F1: {results_a["macro_f1"]:.4f}')

    # === Version B: 有剂量特征 ===
    print('\n' + '=' * 60)
    print('VERSION B: With dosage features')
    print('=' * 60)
    X_b = extract_features(df, herb_dict, include_dosage=True)
    cat_idx_b = [list(X_b.columns).index('urgency')]
    results_b, model_b = train_and_eval(X_b, y_labels, cat_idx_b)

    print(f'Top-1: {results_b["top1"]:.4f}  Top-3: {results_b["top3"]:.4f}  '
          f'Top-5: {results_b["top5"]:.4f}  MRR: {results_b["mrr"]:.4f}')
    print(f'Micro-F1: {results_b["micro_f1"]:.4f}  Macro-F1: {results_b["macro_f1"]:.4f}')

    # === 对比 ===
    print('\n' + '=' * 60)
    print('COMPARISON: A (no dose) vs B (with dose)')
    print('=' * 60)
    print(f'{"Metric":<15} {"A (no dose)":>12} {"B (with dose)":>14} {"Delta":>10}')
    print('-' * 55)
    for key in ['top1','top3','top5','mrr','micro_f1','macro_f1']:
        delta = results_b[key] - results_a[key]
        sign = '+' if delta > 0 else ''
        print(f'{key:<15} {results_a[key]:>12.4f} {results_b[key]:>14.4f} {sign}{delta:>9.4f}')

    # Per-class comparison
    print('\nPer-class F1 comparison:')
    print(f'{"Form":<10} {"A (no dose)":>12} {"B (with dose)":>14} {"Delta":>10}')
    print('-' * 50)
    all_forms = sorted(set(list(results_a['per_class'].keys()) + list(results_b['per_class'].keys())))
    for f in all_forms:
        fa = results_a['per_class'].get(f, 0)
        fb = results_b['per_class'].get(f, 0)
        delta = fb - fa
        if fa > 0 or fb > 0:
            sign = '+' if delta > 0 else ''
            marker = ' <<' if abs(delta) > 0.05 else ''
            print(f'{f:<10} {fa:>12.4f} {fb:>14.4f} {sign}{delta:>9.4f}{marker}')

    # Feature importance from Version B
    print('\nTop 10 features (Version B):')
    importance = model_b.get_feature_importance()
    feat_names = X_b.columns.tolist()
    top10 = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:10]
    for i, (name, imp) in enumerate(top10, 1):
        dosage_mark = ' [DOSAGE]' if name in ['total_weight','avg_dose','max_dose','min_dose','dose_std','dose_range','max_min_ratio','log_total_weight'] else ''
        print(f'  {i:>2}. {name:<30} {imp:>8.2f}{dosage_mark}')

    print('\nDONE')
