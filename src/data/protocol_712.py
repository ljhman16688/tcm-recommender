# -*- coding: utf-8 -*-
"""712 协议数据构建共享模块（2026-10-12 从 compare_kg_leakage_712.py 逐字抽出）。

统一数据口径的唯一来源：药典 + 部颁成方制剂 → records（19 类）
→ 7:1:2 分层随机划分（random_state=42，稀有类<3 归桶）。
compare_kg_leakage_712.py 与 train_bert.py（诚实版）都必须从本模块取数据，
保证 KG 实验与 BERT 实验的样本、标签、划分完全对齐。

与 2026-10-13 P2 侧改造同步的修正（已在 712 脚本同步）：
  - MANUAL_MAP 无条件覆盖：infer_form 已给出剂型也以人工金标为准。
    探针核实（outputs/_probe_manual_map.py）：零剔除、仅 4 处重新标注
    （参贝陈皮/脉络通/小儿疳积糖/痔疮外洗药 由制剂工艺规则误标改正为金标），
    总数保持 5105，训练 3573 / 验证 511 / 测试 1021。
  - 自检（python -u src/data/protocol_712.py）应与 712 脚本重跑输出的
    药典/成方制剂/总计/划分 四个数字完全一致。
"""
import os, sys, re, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.features.extract_features import normalise_herb, ALL_FORMS

with open(os.path.join(_ROOT, 'data', 'herb_property_dict_merged.json'), 'r', encoding='utf-8') as f:
    herb_dict = json.load(f)

FORM_MERGE = {'软胶囊': '胶囊剂', '滴丸': '丸剂', '贴膏剂': '膏剂', '口服液': '合剂'}


def parse_rx(text):
    if pd.isna(text):
        return [], []
    herbs, doses = [], []
    pat = r'([一-龥]{1,6}(?:[（(][^）)]+[）)])?)\s*(\d+\.?\d*)\s*(?:g|ml|mg|克|毫升|毫克)?'
    for m in re.findall(pat, str(text)):
        name = m[0].strip()
        if len(name) >= 2:
            herbs.append(name)
            doses.append(float(m[1]))
    return herbs, doses


def infer_form(name, process):
    name = str(name) if pd.notna(name) else ''
    process = str(process) if pd.notna(process) else ''
    rules = [('滴丸', '丸剂'), ('软胶囊', '胶囊剂'), ('注射液', '注射剂'), ('口服液', '合剂'),
             ('吸入剂', '气雾剂'), ('喷雾剂', '气雾剂'), ('滴眼剂', '洗剂'), ('滴剂', '洗剂'),
             ('糖浆', '糖浆剂'), ('颗粒', '颗粒剂'), ('冲剂', '颗粒剂'), ('胶囊', '胶囊剂'),
             ('片', '片剂'), ('栓', '栓剂'), ('膏', '膏剂'), ('丹', '丹剂'),
             ('散', '散剂'), ('丸', '丸剂'), ('酒', '酒剂'), ('酊', '酊剂'),
             ('茶', '茶剂'), ('露', '露剂'), ('膜', '膜剂'), ('气雾', '气雾剂'),
             ('贴膏', '膏剂'), ('搽剂', '搽剂'), ('洗剂', '洗剂'),
             ('合剂', '合剂'), ('锭', '丹剂'), ('精', '合剂'), ('浆', '糖浆剂'),
             ('蜜', '膏剂'), ('滋', '合剂'), ('灵', '颗粒剂'), ('油', '搽剂'),
             ('液', '合剂'), ('汁', '合剂'), ('饮', '合剂'), ('汤', '合剂'),
             ('曲', '茶剂'), ('胶', '膏剂'), ('粉', '散剂'), ('墨', '丹剂'),
             ('条', '膏剂'), ('绒', '膏剂'), ('乳', '搽剂'), ('净', '搽剂'),
             ('贴', '膏剂'), ('宁', '合剂'), ('糕', '茶剂'), ('雪', '散剂'),
             ('末', '散剂'), ('晶', '颗粒剂'), ('水', '酊剂')]
    for kw, form in rules:
        if kw in name:
            return form
    p_rules = [(r'制粒|整粒', '颗粒剂'), (r'装.*胶囊', '胶囊剂'), (r'压.*片', '片剂'),
               (r'(?:制成|泛).*丸', '丸剂'), (r'制成.*锭', '丹剂'), (r'模成块', '茶剂'),
               (r'分装', '散剂'), (r'灌装|灌封', '合剂'), (r'涂布|涂膜', '膏剂'),
               (r'发酵', '茶剂'), (r'制成.*1000ml', '合剂'),
               (r'粉碎成细粉.*混匀.*即得', '散剂'), (r'加入凡士林', '膏剂'),
               (r'渗漉.*乙醇.*即得', '酊剂')]
    for pat, form in p_rules:
        if re.search(pat, process):
            return form
    return None


MANUAL_MAP = {
    '八宝眼药': '洗剂', '沈阳红药': '胶囊剂', '云南白药': '散剂', '海洋胃药': '片剂',
    '京万红': '膏剂', '伤痛舒': '膏剂', '桂林西瓜霜': '散剂', '儿康宁': '合剂',
    '鼻通宁滴剂': '洗剂', '口腔炎喷雾剂': '气雾剂', '烧伤净喷雾剂': '气雾剂',
    '鼻炎滴剂': '洗剂', '心舒静吸入剂': '气雾剂', '按摩乳': '搽剂',
    '伤可贴': '膏剂', '筋骨止痛凝胶': '凝胶剂', '复方铁苋止血粉': '散剂',
    '药墨': '丹剂', '珍珠层粉': '散剂', '顽癣净': '搽剂', '伤友擦剂': '搽剂',
    '蛇胆姜粒': '茶剂', '橙皮': '茶剂', '药制橄榄盐': '茶剂', '维血宁': '合剂',
    '引阳索': '颗粒剂', '安神宁': '合剂', '生乳汁': '合剂', '升血调元汤': '合剂',
    '川贝半夏液': '合剂', '岩果止咳液': '合剂', '云南蛇药': '合剂',
    '康复新液': '合剂', '保宁半夏曲': '茶剂', '闽东建曲': '茶剂',
    '老范志万应神曲': '茶剂', '六神曲': '茶剂', '小儿疳积糖': '颗粒剂',
    '牙痛药水': '酊剂', '十滴水': '酊剂', '风火眼药': '洗剂', '蚕茧眼药': '洗剂',
    '痔疮外洗药': '洗剂', '白敬宇眼药': '膏剂', '绿雪': '散剂', '口疳吹药': '散剂',
    '吊筋药': '散剂', '健脾八珍糕': '茶剂', '伤科敷药': '散剂', '珍珠八宝眼药': '洗剂',
    '婴儿素': '散剂', '腹痛水': '酊剂', '赛空青眼药': '洗剂', '藿香水': '酊剂',
    '宝宝乐': '颗粒剂', '复方蛇胆陈皮末': '散剂', '八角茴香水': '酊剂',
    '止痒消炎水': '酊剂', '参贝陈皮': '茶剂', '参耳五味晶': '颗粒剂',
    '湛江蛇药': '散剂', '六神祛暑水': '酊剂', '正骨水': '酊剂', '夏天无眼药水': '洗剂',
    '西瓜霜': '散剂', '复方香薷水': '酊剂', '脉络通': '片剂', '蓝花药': '丸剂',
    '清艾条': '膏剂', '清艾绒': '膏剂',
}


def ind_keywords(ind_text):
    if not ind_text:
        return []
    return re.findall(r'[一-龥]{2,6}', str(ind_text))[:10]


def text_of(rec):
    """BERT/LLM 共用的处方文本（药材 + 剂量 + 适应症前 200 字）。
    train_bert.py 诚实版内含一份逐字相同的本地副本（2026-10-13 该脚本首次运行时
    已从旧版上线，为免改跑动中进程未动它，后续可改为从这里导入）。"""
    parts = []
    for h, d in zip(rec['herbs'], rec['doses']):
        parts.append('%s %sg' % (h, d))
    ind = rec['indication'][:200] if rec['indication'] else ''
    return '药材：%s。适应症：%s' % ('、'.join(parts), ind)


def build_records(verbose=True):
    """构建全部 records（药典 + 成方制剂），与 712 协议一致。

    含 2026-10-13 MANUAL_MAP 无条件覆盖修正（原 712 只在 infer_form=None 时套用
    金标，导致「筋骨止痛凝胶」被规则误判为膏剂；金标 凝胶剂 不在 19 类 → 该条剔除）。
    """
    records = []
    df_yp = pd.read_excel(os.path.join(_ROOT, 'data', 'external', '药典方剂数据.xlsx'))
    for _, r in df_yp.iterrows():
        herbs, doses = parse_rx(r['prescription_raw'])
        if not herbs:
            continue
        form = str(r['dosage_form']).strip()
        form = FORM_MERGE.get(form, form)
        if form not in ALL_FORMS:
            continue
        ind = str(r['indication']) if pd.notna(r['indication']) else ''
        admin = str(r['administration']) if pd.notna(r['administration']) else ''
        records.append({'name': str(r['drug_name']).strip(), 'herbs': herbs, 'doses': doses,
                        'indication': ind, 'admin': admin, 'form': form, 'ind_kws': ind_keywords(ind),
                        'source': '药典'})
    if verbose:
        print('  药典: %d 条' % len(records))

    n_yp = len(records)
    df_s1 = pd.read_excel(os.path.join(_ROOT, 'data', 'external', '中药成方制剂数据.xlsx'), sheet_name='Sheet1')
    for _, r in df_s1.iterrows():
        name = str(r['复方名称']).strip() if pd.notna(r['复方名称']) else ''
        if not name or name == 'nan':
            continue
        form = infer_form(r['复方名称'], r['制法'])
        if name in MANUAL_MAP:
            form = MANUAL_MAP[name]
        if form is None or form not in ALL_FORMS:
            continue
        herbs, doses = parse_rx(r['含量/克数'])
        if not herbs:
            herbs, _ = parse_rx(r['处方'])
        if not herbs:
            continue
        ind = str(r['功能与主治']) if pd.notna(r['功能与主治']) else ''
        admin = str(r['用量']) if pd.notna(r['用量']) else ''
        records.append({'name': name, 'herbs': herbs, 'doses': doses,
                        'indication': ind, 'admin': admin, 'form': form, 'ind_kws': ind_keywords(ind),
                        'source': '成方制剂'})
    if verbose:
        print('  成方制剂: %d 条' % (len(records) - n_yp))
        print('  总计: %d 条' % len(records))
    return records


def split_records(records, verbose=True):
    """7:1:2 分层随机划分（random_state=42），返回 dict：
    train/val/test（record 子集）+ train_idx/val_idx/test_idx + y + le。"""
    le = LabelEncoder()
    le.fit(sorted(set(r['form'] for r in records)))
    y = le.transform([r['form'] for r in records])

    # 稀有类别(<3)在分层时归入一个桶，避免 sklearn 分层失败
    def _strat(yy):
        ys = yy.copy()
        cnt = pd.Series(ys).value_counts()
        for rc in cnt[cnt < 3].index:
            ys[ys == rc] = -1
        return ys

    idx = np.arange(len(records))
    trainval_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=_strat(y), random_state=42)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.125, stratify=_strat(y[trainval_idx]), random_state=42)
    if verbose:
        print('  划分: 训练 %d / 验证 %d / 测试 %d' % (len(train_idx), len(val_idx), len(test_idx)))
    return {'train': [records[i] for i in train_idx],
            'val': [records[i] for i in val_idx],
            'test': [records[i] for i in test_idx],
            'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
            'y': y, 'le': le}


def handcrafted_features(herbs, doses, indication, admin=''):
    """68 维手工特征（与 compare_kg_leakage_712.py 逐字一致）。"""
    n = len(herbs)
    feats = {'num_herbs': n}
    nature_fold = {'微寒': '寒', '大寒': '寒', '微温': '温', '大热': '热'}
    taste_fold = {'微苦': '苦', '微甘': '甘', '微辛': '辛'}
    nc = {'寒': 0, '凉': 0, '平': 0, '温': 0, '热': 0}
    tc = {'辛': 0, '甘': 0, '酸': 0, '苦': 0, '咸': 0, '淡': 0, '涩': 0}
    mc = {'心': 0, '肝': 0, '脾': 0, '肺': 0, '肾': 0, '胃': 0,
          '胆': 0, '小肠': 0, '大肠': 0, '膀胱': 0, '三焦': 0, '心包': 0}
    xc = {'hard': 0, 'starchy': 0, 'fibrous': 0, 'mucilaginous': 0,
          'oily': 0, 'gelatinous': 0, 'normal': 0}
    cv = ct = cm = cp = kn = 0
    for herb in herbs:
        hc = normalise_herb(herb)
        props = herb_dict.get(hc) or herb_dict.get(herb)
        if props is None:
            xc['normal'] += 1
            continue
        kn += 1
        if props.get('volatile'): cv += 1
        if props.get('toxic'): ct += 1
        if props.get('mineral'): cm += 1
        if props.get('precious'): cp += 1
        nat = props.get('nature', '')
        nat = nature_fold.get(nat, nat)
        if nat in nc: nc[nat] += 1
        for t in props.get('taste', []):
            t = taste_fold.get(t, t)
            if t in tc: tc[t] += 1
        for m in props.get('meridian', []):
            if m in mc: mc[m] += 1
        tex = props.get('texture', 'normal')
        if tex in xc: xc[tex] += 1
    feats['has_volatile'] = int(cv > 0)
    feats['has_toxic'] = int(ct > 0)
    feats['has_mineral'] = int(cm > 0)
    feats['has_precious'] = int(cp > 0)
    feats['volatile_ratio'] = cv / n
    feats['toxic_ratio'] = ct / n
    feats['mineral_ratio'] = cm / n
    feats['precious_ratio'] = cp / n
    feats['known_ratio'] = kn / n
    for nat in ['寒', '凉', '平', '温', '热']:
        feats['nature_' + nat + '_ratio'] = nc[nat] / n
    for t in ['辛', '甘', '酸', '苦', '咸', '淡', '涩']:
        feats['taste_' + t + '_ratio'] = tc[t] / n
    for m in ['心', '肝', '脾', '肺', '肾', '胃', '胆', '小肠', '大肠', '膀胱', '三焦', '心包']:
        feats['meridian_' + m + '_ratio'] = mc[m] / n
    for tex in ['hard', 'starchy', 'fibrous', 'mucilaginous', 'oily', 'gelatinous']:
        feats['texture_' + tex + '_ratio'] = xc[tex] / n

    ind = str(indication) if indication else ''
    if any(w in ind for w in ['急症', '暴发', '急性']):
        feats['urgency'] = 'acute'
    elif any(w in ind for w in ['慢性', '久', '迁延']):
        feats['urgency'] = 'chronic'
    else:
        feats['urgency'] = 'unspecified'
    admin = str(admin) if admin else ''
    feats['is_topical'] = int(any(w in admin for w in ['外用', '涂', '敷', '贴', '搽', '洗', '熏', '点眼', '滴眼', '滴鼻', '滴耳', '灌肠', '注入', '塞入', '塞肛', '阴道', '吹', '喷', '滴入']))
    for k, ps in {
        'syndrome_blood_stasis': ['血瘀', '化瘀', '活血'],
        'syndrome_qi_deficiency': ['气虚', '益气', '补气'],
        'syndrome_blood_deficiency': ['血虚', '补血', '养血'],
        'syndrome_yin_deficiency': ['阴虚', '滋阴', '养阴'],
        'syndrome_yang_deficiency': ['阳虚', '温阳', '补阳', '壮阳'],
        'syndrome_damp_heat': ['湿热', '清热利湿', '利湿', '化湿'],
        'syndrome_wind': ['风寒', '风热', '风湿', '祛风', '疏风'],
        'syndrome_phlegm': ['化痰', '祛痰', '豁痰'],
        'syndrome_fire_toxin': ['热毒', '火毒', '清热解毒', '泻火'],
        'syndrome_qi_stagnation': ['气滞', '理气', '行气', '疏肝'],
    }.items():
        feats[k] = int(any(p in ind for p in ps))
    for k, ps in {
        'disease_respiratory': ['感冒', '咳嗽', '喘', '哮', '咽', '喉'],
        'disease_digestive': ['胃', '肠', '胆', '泻', '秘', '食积'],
        'disease_cardiovascular': ['胸痹', '心悸', '眩晕', '中风'],
        'disease_gynecology': ['月经', '带下', '崩漏', '胎'],
        'disease_rheumatology': ['风湿', '痹', '筋骨', '关节'],
        'disease_dermatology': ['疮', '痈', '疹', '癣'],
        'disease_pediatrics': ['小儿', '惊风'],
        'disease_urology': ['淋', '浊', '水肿'],
    }.items():
        feats[k] = int(any(p in ind for p in ps))
    if doses:
        a = np.array(doses)
        feats['total_weight'] = a.sum()
        feats['avg_dose'] = a.mean()
        feats['max_dose'] = a.max()
        feats['min_dose'] = a.min()
        feats['dose_std'] = a.std()
        feats['dose_range'] = a.max() - a.min()
        feats['max_min_ratio'] = a.max() / a.min() if a.min() > 0 else 0
        feats['log_total_weight'] = np.log1p(a.sum())
    else:
        for k in ['total_weight', 'avg_dose', 'max_dose', 'min_dose',
                  'dose_std', 'dose_range', 'max_min_ratio', 'log_total_weight']:
            feats[k] = 0
    return feats


if __name__ == '__main__':
    recs = build_records()
    split = split_records(recs)
    n_tr = len(split['train_idx']); n_va = len(split['val_idx']); n_te = len(split['test_idx'])
    n_cls = len(split['le'].classes_)
    # 探针核实（outputs/_probe_manual_map.py，2026-10-13）：MANUAL_MAP 修正零剔除、
    # 仅 4 处重新标注（参贝陈皮/脉络通/小儿疳积糖/痔疮外洗药），总数保持 5105。
    ok = (len(recs) == 5105 and n_cls == 19
          and n_tr + n_va + n_te == len(recs)
          and split['y'][split['test_idx']].shape[0] == n_te)
    print('自检:', 'PASS' if ok else 'FAIL',
          '(总数 %d / 训练 %d / 验证 %d / 测试 %d / %d 类)' % (len(recs), n_tr, n_va, n_te, n_cls))