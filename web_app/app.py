# -*- coding: utf-8 -*-
"""
中药剂型智能推荐 Web 应用
Flask 后端：特征提取 + CatBoost 推理 + KG 锚定 + 模板推理
启动: python web_app/app.py
"""

import sys, io, re, json, os
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, jsonify

# 确保项目根目录在 path 中，方便加载 data/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.features.extract_features import (
    normalise_herb, parse_prescription, FeatureExtractor,
    ALL_FORMS, EXTRACT_MAP, ALIAS,
)
from src.features.llm_herb_property import infer_herb_properties

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# ==============================================
# 全局资源（启动时加载）
# ==============================================
print('Loading KG-RAR resources...')

with open('data/herb_property_dict.json', 'r', encoding='utf-8') as f:
    herb_dict = json.load(f)
with open('data/tcm_knowledge_graph.json', 'r', encoding='utf-8') as f:
    kg = json.load(f)
with open('data/dosage_form_rules.json', 'r', encoding='utf-8') as f:
    rules_db = json.load(f)['rules']

print(f'  Rules: {len(rules_db)} rules loaded from dosage_form_rules.json')

# 初始化特征提取器
extractor = FeatureExtractor(herb_dict, kg)

# 剂型列表（从 extract_features 模块导入 ALL_FORMS）

# KG 索引
id2ent = {int(eid): info['name'] for eid, info in kg['entities'].items()}
herb_to_formulas = {}
formula_to_form = {}
formula_to_name = {}
formula_to_herbs = {}  # 处方ID → herb names

for t in kg['triples']:
    if t['relation_name'] == 'contained_in':
        herb_to_formulas.setdefault(t['head'], []).append(t['tail'])
    elif t['relation_name'] == 'has_form':
        formula_to_form[t['head']] = t['tail_name'].replace('dosage_form:', '')
        formula_to_name[t['head']] = t['head_name']

# 构建公式→药材反向索引
for t in kg['triples']:
    if t['relation_name'] == 'contained_in':
        fid = t['tail']
        if fid not in formula_to_herbs:
            formula_to_herbs[fid] = set()
        formula_to_herbs[fid].add(id2ent.get(t['head'], ''))

# 药材→剂型偏好（复用 extractor 的索引）
herb_name_to_forms = extractor.herb_name_to_forms
ind_to_forms = extractor.ind_to_forms

# 统计 KG 中实际可用的药材数（标准化后在 herb_dict 中有属性的）
_kg_herbs_normalized = set()
for h_name in herb_name_to_forms:
    _kg_herbs_normalized.add(normalise_herb(h_name))
_kg_herbs_known = sum(1 for h in _kg_herbs_normalized if h in herb_dict)
print(f'  KG: {len(formula_to_form)} formulas, '
      f'{len(herb_name_to_forms)} herb entities '
      f'(→ {len(_kg_herbs_normalized)} unique, {_kg_herbs_known} in herb_dict)')

# extract_features 函数引用（兼容旧代码）
extract_features = extractor.extract_features

# ==============================================
# ==============================================


# ==============================================
# 训练/加载 CatBoost 模型（首次训练后保存 .cbm，后续秒加载）
# ==============================================
print('Preparing CatBoost model...')

from sklearn.preprocessing import LabelEncoder
import pickle

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CBM_PATH = os.path.join(MODEL_DIR, 'catboost_model.cbm')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
COLS_PATH = os.path.join(MODEL_DIR, 'feature_cols.pkl')

if os.path.exists(CBM_PATH) and os.path.exists(ENCODER_PATH):
    print('  Loading saved model...')
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(CBM_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    with open(COLS_PATH, 'rb') as f:
        feat_cols = pickle.load(f)
    print(f'  Model loaded: {len(feat_cols)} features, {len(le.classes_)} classes')
else:
    print('  Training new model (one-time, ~30s)...')
    # 加载预计算特征表
    X1 = pd.read_csv('outputs/datasets/Dataset1_pharmacopoeia_features.csv')
    X2 = pd.read_csv('outputs/datasets/Dataset2_ministerial_features.csv')
    df_all = pd.concat([X1, X2], ignore_index=True)

    form_counts = df_all['form'].value_counts()
    keep = form_counts[form_counts >= 5].index
    df_all = df_all[df_all['form'].isin(keep)].copy()

    le = LabelEncoder()
    y = le.fit_transform(df_all['form'])

    meta = ['drug_name', 'herb_list', 'indication_text', 'form']
    feat_cols = [c for c in df_all.columns if c not in meta]
    X = df_all[feat_cols].fillna(0).copy()
    if 'urgency' in X.columns:
        urg_map = {'acute': 0, 'chronic': 1, 'unspecified': 2}
        X['urgency'] = X['urgency'].map(urg_map).fillna(2)

    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        random_seed=42, loss_function='MultiClass', verbose=0
    )
    model.fit(X, y)
    model.save_model(CBM_PATH)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    with open(COLS_PATH, 'wb') as f:
        pickle.dump(feat_cols, f)
    print(f'  Model trained & saved: {len(X)} samples, {len(le.classes_)} classes')


# ==============================================
# 规则引擎：25条剂型选择规则，4级证据体系
# ==============================================

def evaluate_trigger(trigger, herb_list, features):
    """评估单条规则的触发条件"""
    ttype = trigger['type']
    # 药材匹配
    if ttype == 'herb_match':
        target_herbs = set(trigger['herbs'])
        mode = trigger.get('match_mode', 'any')
        matched = [h for h in herb_list if normalise_herb(h) in target_herbs or h in target_herbs]
        return (len(matched) > 0, matched) if mode == 'any' else (len(matched) == len(target_herbs), matched)
    # 药材属性检查
    elif ttype == 'property_check':
        field = trigger['field']
        if field in ('has_volatile', 'has_toxic', 'has_mineral', 'has_precious'):
            hkey = field.replace('has_', '')
            def _get_prop(h):
                hc = normalise_herb(h)
                p = herb_dict.get(hc) or herb_dict.get(h)
                if p is None:
                    p = infer_herb_properties(h) or {}
                return p.get(hkey)
            return (any(_get_prop(h) for h in herb_list), [])
        return (features.get(field, 0) > 0, [])
    # 质地比例检查
    elif ttype == 'texture_check':
        return (features.get(trigger['field'], 0) > trigger.get('value', 0), [])
    # 特征值检查
    elif ttype == 'feature_check':
        fv = features.get(trigger['field'])
        if fv is None:
            return (False, [])
        op = trigger.get('op', 'eq')
        return (fv == trigger['value'], []) if op == 'eq' else (fv >= trigger['value'], [])
    # 复合条件
    elif ttype == 'composite':
        results = []
        for c in trigger['conditions']:
            if c.get('type') == 'form_check':
                # form_check 是特殊条件，在外面处理（用于 L1-002 的复合规则）
                results.append(True)
            else:
                results.append(evaluate_trigger(c, herb_list, features)[0])
        return (all(results), []) if trigger['op'] == 'and' else (any(results), [])
    return (False, [])


def build_features(herb_list, indication):
    """构建规则引擎所需的特征"""
    features = {'num_herbs': len(herb_list)}
    n = len(herb_list)
    if n > 0:
        xc = {'starchy': 0, 'fibrous': 0, 'mucilaginous': 0, 'oily': 0, 'gelatinous': 0, 'mineral': 0}
        for herb in herb_list:
            nh = normalise_herb(herb)
            props = herb_dict.get(nh) or herb_dict.get(herb)  # 优先标准化名
            if props is None:
                props = infer_herb_properties(herb) or {}
            t = props.get('texture', 'normal')
            if t in xc:
                xc[t] += 1
            # 矿物药计数：通过 has_mineral 属性判断
            is_mineral_by_prop = props.get('mineral') or props.get('has_mineral')
            if is_mineral_by_prop:
                xc['mineral'] += 1
        features['starchy_ratio'] = xc['starchy'] / n
        features['fibrous_ratio'] = xc['fibrous'] / n
        features['mucilaginous_ratio'] = xc['mucilaginous'] / n
        features['oily_ratio'] = xc['oily'] / n
        features['gelatinous_ratio'] = xc['gelatinous'] / n
        features['mineral_ratio'] = xc['mineral'] / n
        features['total_weight'] = 0
        ind = str(indication) if indication else ''
        # 急慢性判断：显式关键词 > 隐式疾病模式
        acute_explicit = ['急症', '暴发', '急性', '骤发', '突发', '高热', '惊厥', '抽搐', '神昏', '谵语', '中毒', '外伤']
        chronic_explicit = ['慢性', '久病', '迁延', '日久', '久治', '劳损', '痿证', '不孕', '不育', '消渴']
        # 隐式慢性：特定疾病模式本身暗示慢性病程
        chronic_implicit = ['月经', '带下', '崩漏', '胎动', '虚', '亏', '损', '痹', '风湿', '筋骨', '关节', '腰膝', '耳鸣', '健忘', '失眠', '遗精', '阳痿', '早泄', '自汗', '盗汗', '乏力', '纳差', '消瘦', '癥瘕', '积聚', '瘿瘤', '瘰疬', '心悸', '眩晕', '胸痹', '怔忡']
        if any(w in ind for w in acute_explicit):
            features['urgency'] = 'acute'
        elif any(w in ind for w in chronic_explicit):
            features['urgency'] = 'chronic'
        elif any(w in ind for w in chronic_implicit):
            features['urgency'] = 'chronic'
        else:
            features['urgency'] = 'unspecified'
        features['is_topical'] = int(any(w in ind for w in ['外用', '涂', '敷', '贴']))
        disease_kw = {
            'disease_respiratory': ['感冒', '咳嗽', '喘', '哮', '咽', '喉'],
            'disease_digestive': ['胃', '肠', '胆', '泻', '秘', '食积'],
            'disease_cardiovascular': ['胸痹', '心悸', '眩晕', '中风'],
            'disease_gynecology': ['月经', '带下', '崩漏', '胎'],
            'disease_rheumatology': ['风湿', '痹', '筋骨', '关节'],
            'disease_dermatology': ['疮', '痈', '疹', '癣'],
            'disease_pediatrics': ['小儿', '惊风'],
            'disease_urology': ['淋', '浊', '水肿'],
        }
        for k, ps in disease_kw.items():
            features[k] = int(any(p in ind for p in ps))
    return features


def match_rules(herb_list, features):
    """匹配全部规则，按证据等级分组返回"""
    matched = {'L1': [], 'L2': [], 'L3': [], 'L4': []}
    for rule in rules_db:
        triggered, matched_herbs = evaluate_trigger(rule['trigger'], herb_list, features)
        if triggered:
            rec = rule['recommendation']
            if matched_herbs:
                rec = rec.replace('{herb}', '、'.join(matched_herbs[:3]))
            level_key = 'L%d' % rule['evidence_level']
            matched[level_key].append({
                'id': rule['id'],
                'rule_name': rule['rule_name'],
                'category': rule['category'],
                'action': rule['action'],
                'target_forms': rule.get('target_forms', []),
                'boost_weight': rule.get('boost_weight'),
                'recommendation': rec,
                'source': rule['source'],
                'exceptions': rule.get('exceptions', []),
                'expert_consensus': rule.get('expert_consensus', ''),
                'evidence_level': rule['evidence_level'],
                'certainty': rule.get('certainty', ''),
            })
    return matched


def apply_rules(herb_props_detail, indication):
    """兼容旧接口：基于规则引擎返回排除的剂型及原因（带证据等级）"""
    herb_list = [p['name'] for p in herb_props_detail if p.get('found')]
    features = build_features(herb_list, indication)
    all_matched = match_rules(herb_list, features)

    # 合并 L1+L2 的排除/警告为 excluded 列表
    excluded = []
    for level in ['L1', 'L2']:
        for r in all_matched[level]:
            for f in r['target_forms']:
                excluded.append({
                    'form': f,
                    'reason': r['recommendation'],
                    'evidence_level': r['evidence_level'],
                    'rule_name': r['rule_name'],
                    'source': r['source'],
                })

    # 去重（同剂型只保留最高证据等级）
    seen = {}
    unique = []
    for e in excluded:
        if e['form'] not in seen or e['evidence_level'] < seen[e['form']]:
            seen[e['form']] = e['evidence_level']
            unique = [x for x in unique if x['form'] != e['form']]
            unique.append(e)
    return unique, all_matched


# ==============================================
# KG 检索
# ==============================================
def retrieve_kg_evidence(herbs, indication):
    """四路 KG 检索"""
    evidence = {
        'herb_props': [],
        'similar_formulas': [],
        'disease_preference': {},
        'rules': []
    }

    # 路①: 药材属性
    for herb in herbs:
        hc = normalise_herb(herb)
        props = herb_dict.get(hc) or herb_dict.get(herb)  # 优先标准化名
        if props is None:
            # LLM 兜底：字典中查不到时，调用 LLM 推断属性
            props = infer_herb_properties(herb)
        if props:
            kg_prefs = herb_name_to_forms.get(herb, herb_name_to_forms.get('herb:' + herb, {}))
            top_forms = sorted(kg_prefs.items(), key=lambda x: -x[1])[:3]
            evidence['herb_props'].append({
                'name': herb,
                'nature': props.get('nature', ''),
                'taste': props.get('taste', []),
                'meridian': props.get('meridian', []),
                'kg_top_forms': [{'form': f, 'ratio': round(p * 100)} for f, p in top_forms],
                '_source': props.get('_source', 'dict'),
            })
        else:
            evidence['herb_props'].append({
                'name': herb, 'found': False
            })

    # 路②: 相似处方（药材重叠）
    input_herbs_set = set(normalise_herb(h) for h in herbs)
    similar = []
    for fid, f_herbs in formula_to_herbs.items():
        if len(f_herbs) < 2:
            continue
        overlap = input_herbs_set & f_herbs
        if len(overlap) >= 2:
            jaccard = len(overlap) / len(input_herbs_set | f_herbs)
            if jaccard >= 0.3:
                fname = formula_to_name.get(fid, '未知方')
                fform = formula_to_form.get(fid, '未知')
                similar.append({
                    'name': fname.replace('formula:', ''),
                    'form': fform,
                    'overlap': list(overlap),
                    'overlap_count': len(overlap),
                    'jaccard': round(jaccard, 2)
                })
    similar.sort(key=lambda x: (-x['overlap_count'], -x['jaccard']))
    evidence['similar_formulas'] = similar[:5]

    # 路③: 疾病-剂型偏好
    ind_kw = re.findall(r'[一-龥]{2,6}', str(indication) if indication else '')
    disease_forms = Counter()
    for kw in ind_kw[:10]:
        for f in ALL_FORMS:
            disease_forms[f] += ind_to_forms.get((kw, f), 0)
    total = sum(disease_forms.values())
    if total > 0:
        evidence['disease_preference'] = {
            f: round(disease_forms[f] / total * 100) for f in ALL_FORMS
            if disease_forms[f] > 0
        }

    return evidence


# ==============================================
# 模板推理引擎
# ==============================================
def generate_reasoning(top5, evidence, excluded, herb_props_detail, all_matched, anchor_match=None):
    """基于 KG 证据和规则引擎生成推理文本"""
    lines = []

    # 首推理由
    top1 = top5[0]

    # KG 锚定（精确药材集匹配）
    if anchor_match:
        lines.append(f'KG锚定识别：该处方与已知方「{anchor_match["name"]}」药材组成完全一致，其标准剂型为**{anchor_match["form"]}**，直接置顶。')
    else:
        lines.append(f'首选推荐**{top1["form"]}**（置信度 {top1["confidence"]:.1%}）。')

    # 规则引擎证据（按等级）
    level_labels = {1: '法规禁忌', 2: '风险警示', 3: '工艺调整建议', 4: '弱先验'}
    for level in [1, 2, 3, 4]:
        lr = all_matched.get('L%d' % level, [])
        if lr:
            for r in lr[:3]:  # 每级最多3条
                lines.append(f'[L{level} {level_labels[level]}] {r["recommendation"]}')

    # 药材属性证据
    volatile_herbs = [p['name'] for p in herb_props_detail if p.get('volatile')]
    mineral_herbs = [p['name'] for p in herb_props_detail if p.get('mineral')]
    precious_herbs = [p['name'] for p in herb_props_detail if p.get('precious')]
    toxic_herbs = [p['name'] for p in herb_props_detail if p.get('toxic')]

    if mineral_herbs:
        lines.append(f'处方含矿物药{"、".join(mineral_herbs)}，质地坚硬难溶于水，宜制成丸散剂。')
    if precious_herbs:
        lines.append(f'含贵重药{"、".join(precious_herbs)}，应避免煎煮损失，宜散剂或胶囊剂直接入药。')
    if volatile_herbs:
        lines.append(f'含挥发性成分{"、".join(volatile_herbs)}，不宜久煎，推荐采用适宜保留挥发性成分的剂型。')
    if toxic_herbs:
        lines.append(f'含毒性药材{"、".join(toxic_herbs)}，严禁注射剂，需严格控制剂量。')

    # KG 历史偏好
    top_herb_forms = Counter()
    for p in evidence['herb_props']:
        if p.get('found', True) and 'kg_top_forms' in p:
            for f in p['kg_top_forms']:
                top_herb_forms[f['form']] += f['ratio']
    if top_herb_forms:
        top3 = top_herb_forms.most_common(3)
        lines.append(f'从历史处方统计看，方中各药材最常采用的剂型为：{"、".join(f"{f}({r}%)" for f, r in top3)}。')

    # 相似处方
    if evidence['similar_formulas']:
        best = evidence['similar_formulas'][0]
        lines.append(
            f'知识图谱中检索到相似处方「{best["name"]}」（药材重叠 {best["overlap_count"]} 味：{"、".join(best["overlap"])}），其标准剂型为{best["form"]}。')

    return '\n\n'.join(lines)


# ==============================================
# LLM 增强推理（KG-RAG 的 Generate 环节，体现大语言模型）
# ==============================================
LLM_API_KEY = 'sk-zk2faa63c8d59e691b199fcdb2873b1624569afdfb598afe'
LLM_BASE_URL = 'https://api.zhizengzeng.com/v1'
LLM_MODEL = 'qwen3.7-max'

LLM_SYSTEM_PROMPT = """你是资深中药制剂专家。系统已通过知识图谱检索和规则引擎（4级证据体系）完成了预推理，你的任务是整合这些信息，为每个推荐剂型生成推理依据，并标注与规则引擎的冲突。

规则引擎的4级证据体系：
- L1 法规禁忌（硬约束）：基于药典/法规，绝对不可违反。若某剂型被L1排除，绝不可推荐。
- L2 风险警示：基于药理学/毒理学证据，需审慎评估。若某剂型被L2警示，需在推理中提及风险。
- L3 工艺调整建议：基于药剂学，建议性指导。若推荐某剂型但L3提出工艺限制，需在推理中说明条件。
- L4 弱先验：基于KG统计规律，弱参考信号。若推荐与L4偏好一致，可作为加分项；若不一致，可忽略。

严格约束：
1. 推荐依据必须严格对应"分类器 Top-5"列出的5个剂型，逐一生成，剂型名称必须与列表完全一致。
2. 推理时必须交叉引用规则引擎结论：若某推荐剂型与L1/L2/L3规则冲突，必须在reason中明确指出冲突。
3. 排除的剂型从"分类器 Top-5"之外的常见剂型中选取，优先排除被L1/L2命中的剂型。
4. 禁止杜撰不存在的药理机制或临床证据。

只输出 JSON（不要输出其他内容）：
{"recommendations":[{"rank":1,"form":"剂型名","reason":"推荐依据（含规则交叉验证）"},{"rank":2,"form":"剂型名","reason":"推荐依据（含规则交叉验证）"},{"rank":3,"form":"剂型名","reason":"推荐依据（含规则交叉验证）"},{"rank":4,"form":"剂型名","reason":"推荐依据（含规则交叉验证）"},{"rank":5,"form":"剂型名","reason":"推荐依据（含规则交叉验证）"}],"excluded":[{"form":"剂型名","reason":"排除原因"}]}"""


def llm_generate_reasoning(top5, evidence, herbs, indication, all_matched):
    """调用大语言模型，基于分类器结果 + KG证据 + 规则引擎 生成可解释推荐"""
    # 分类器 Top-5
    classifier_str = '\n'.join(
        f'{i+1}. {item["form"]}（置信度 {item["confidence"]:.1%}）'
        for i, item in enumerate(top5)
    )
    # 药材属性与 KG 历史偏好
    prop_lines = []
    for p in evidence.get('herb_props', []):
        if not p.get('found', True):
            continue
        tastes = ''.join(p.get('taste', []))
        mers = ''.join(p.get('meridian', []))
        prefs = '、'.join(f'{f["form"]}({f["ratio"]}%)' for f in p.get('kg_top_forms', []))
        prop_lines.append(f'- {p["name"]}：性{p.get("nature", "")}，味{tastes}，归{mers}经；KG历史偏好 {prefs}')
    prop_str = '\n'.join(prop_lines) if prop_lines else '（无）'
    # 相似处方
    sim_lines = []
    for f in evidence.get('similar_formulas', [])[:5]:
        sim_lines.append(f'- {f["name"]}（{f["form"]}，药材重叠 {f["overlap_count"]} 味）')
    sim_str = '\n'.join(sim_lines) if sim_lines else '（无）'
    # 适应症-剂型偏好
    dis_items = sorted(evidence.get('disease_preference', {}).items(), key=lambda x: -x[1])[:5]
    dis_str = '\n'.join(f'- {f}（{pct}%）' for f, pct in dis_items) if dis_items else '（无）'

    # 规则引擎结论
    level_labels = {1: 'L1-法规禁忌（硬约束）', 2: 'L2-风险警示',
                    3: 'L3-工艺调整建议', 4: 'L4-弱先验'}
    rule_lines = []
    for level in [1, 2, 3, 4]:
        lr = all_matched.get('L%d' % level, [])
        if lr:
            for r in lr:
                targets = '、'.join(r.get('target_forms', [])) or '全部'
                rule_lines.append(
                    f'[{level_labels[level]}] {r["rule_name"]} -> 目标剂型: {targets} | '
                    f'{r["recommendation"][:150]}'
                )
    rule_str = '\n'.join(rule_lines) if rule_lines else '（无匹配规则）'

    prompt = f"""【药材组成】{'、'.join(herbs)}
【适应症】{indication or '未填写'}

【分类器 Top-5】
{classifier_str}

【知识图谱检索依据】
药材属性与历史偏好：
{prop_str}
相似处方：
{sim_str}
适应症-剂型偏好：
{dis_str}

【规则引擎结论（4级证据体系）】
{rule_str}

请为上述5个推荐剂型分别生成一句推理依据，并给出应排除的剂型及原因。
特别注意：推理时必须交叉引用上述规则引擎结论，若某推荐与规则冲突，必须在reason中明确指出。"""

    from openai import OpenAI
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{'role': 'system', 'content': LLM_SYSTEM_PROMPT},
                  {'role': 'user', 'content': prompt}],
        temperature=0, max_tokens=2000,
    )
    content = resp.choices[0].message.content or ''
    # 清理推理标签 + markdown 代码块
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        data = json.loads(m.group())
        recs = data.get('recommendations', [])
        excs = data.get('excluded', [])
        # 防线：只保留分类器 Top-5 中真实存在的剂型
        top5_forms = {item['form'] for item in top5}
        lines = []
        conflicts = []
        for r in recs:
            form = r.get('form', '')
            reason = r.get('reason', '')
            if form and reason and form in top5_forms:
                conflict_tags = _check_rule_conflicts(form, reason, all_matched)
                if conflict_tags:
                    conflicts.append({'form': form, 'conflicts': conflict_tags})
                    reason += ' ' + ' '.join(conflict_tags)
                lines.append(f'推荐**{form}**：{reason}')
        reasoning = '\n\n'.join(lines) if lines else None
        excluded = [{'form': e['form'], 'reason': e['reason']}
                    for e in excs
                    if e.get('form') and e.get('reason') and e['form'] not in top5_forms]
        return reasoning, excluded, conflicts
    except Exception:
        return None, None, []


def _detect_rule_conflicts(all_matched):
    """检测规则引擎内部的潜在冲突——不同规则对同一剂型给出了相反的信号"""
    conflicts = []
    # 收集所有规则对剂型的信号方向
    form_signals = {}  # form -> [(rule_name, signal, level)]

    for level_key, rules in all_matched.items():
        level = int(level_key[1])
        for r in rules:
            signal = 'boost' if r['action'] in ('boost_rank', 'suggest_prefer') else (
                'exclude' if r['action'] in ('exclude', 'warn_exclude') else 'warn')
            for f in r.get('target_forms', []):
                if f not in form_signals:
                    form_signals[f] = []
                form_signals[f].append((r['rule_name'], signal, level))

    # 检测同一剂型是否有相反信号
    for form, signals in form_signals.items():
        has_exclude = any(s[1] == 'exclude' for s in signals)
        has_boost = any(s[1] == 'boost' for s in signals)
        has_warn = any(s[1] == 'warn' for s in signals)

        if has_exclude and has_boost:
            exclude_rules = [s[0] for s in signals if s[1] == 'exclude']
            boost_rules = [s[0] for s in signals if s[1] == 'boost']
            conflicts.append({
                'form': form,
                'type': 'exclude_vs_boost',
                'message': f'L1排除规则（{", ".join(exclude_rules)}）与L3/L4偏好规则（{", ".join(boost_rules)}）存在冲突：该剂型被硬约束排除，但统计偏好倾向于推荐。已按L1排除处理。',
                'severity': 'resolved_l1'
            })
        elif has_warn and has_boost:
            warn_rules = [s[0] for s in signals if s[1] == 'warn']
            boost_rules = [s[0] for s in signals if s[1] == 'boost']
            conflicts.append({
                'form': form,
                'type': 'warn_vs_boost',
                'message': f'L2风险警示（{", ".join(warn_rules)}）与L3/L4偏好规则（{", ".join(boost_rules)}）存在分歧：该剂型统计上被偏好，但存在安全风险警示，需综合评估。',
                'severity': 'attention'
            })

    return conflicts


def _check_rule_conflicts(form, reason, all_matched):
    """检查 LLM 推理是否与规则引擎结论冲突，返回冲突标签列表"""
    conflicts = []
    # 1. 抓取 LLM 自述的冲突/不一致（排除"无冲突"等否定表述）
    self_conflict_patterns = [
        ('不一致', '不一致'),
        ('矛盾', '矛盾'),
        ('潜在冲突', '潜在冲突'),
        ('存在轻微', '存在轻微'),
        ('潜在风险', '潜在风险'),
        ('可能加重', '可能加重'),
        ('需慎用', '需慎用'),
        ('不建议', '不建议'),
        ('需注意L2', '需注意L2'),
        ('需注意L3', '需注意L3'),
        ('与L4.*不一致', '与L4不一致'),
        ('与L3存在', '与L3存在'),
    ]
    import re as _re
    for pattern, label in self_conflict_patterns:
        if pattern in reason and '无' + pattern not in reason:
            idx = reason.find(pattern)
            snippet = reason[max(0, idx-5):idx+len(pattern)+25]
            conflicts.append(f'[LLM标注] {snippet}...')
            break

    # 2. L1 硬约束
    for r in all_matched.get('L1', []):
        if form in r.get('target_forms', []):
            conflicts.append(f'[L1冲突] {r["rule_name"]}')

    # 3. L2 风险警示
    for r in all_matched.get('L2', []):
        if form in r.get('target_forms', []):
            risk_kw = ['风险', '警示', '毒性', '安全', '审慎', '监测', '注意',
                       '不宜', '不建议', '谨慎', '慎重', '禁忌']
            if not any(kw in reason for kw in risk_kw):
                conflicts.append(f'[L2风险未提及] {r["rule_name"]}')

    # 4. L3 工艺建议
    for r in all_matched.get('L3', []):
        if form in r.get('target_forms', []) and r['action'] in ('exclude', 'warn_exclude'):
            proc_kw = ['工艺', '调整', '条件', '后下', '短时', '低温', '不宜',
                       '需', '应', '建议', '注意']
            if not any(kw in reason for kw in proc_kw):
                conflicts.append(f'[L3工艺未提及] {r["rule_name"]}')

    return conflicts


# ==============================================
# 预测函数
# ==============================================
def predict(herbs, doses, indication, herb_props_detail):
    feats, _ = extract_features(herbs, doses, indication)
    if feats is None:
        return None

    X_input = pd.DataFrame([feats])
    X_input = X_input.reindex(columns=feat_cols, fill_value=0)
    if 'urgency' in X_input.columns:
        urg_map = {'acute': 0, 'chronic': 1, 'unspecified': 2}
        X_input['urgency'] = X_input['urgency'].map(urg_map).fillna(2)

    probs = model.predict_proba(X_input)[0]

    # ---- 规则引擎：L1硬排除 + L2风险警示 + L3/L4纯解释（不再加权） ----
    herb_list = [p['name'] for p in herb_props_detail if p.get('found')]
    rule_features = build_features(herb_list, indication)
    all_matched = match_rules(herb_list, rule_features)

    # 记录哪些剂型被规则加权 / 被L2警示
    rule_evidence = set()   # 有 L3/L4 规则支持的剂型（解释性，不改变概率）
    l2_warned_forms = {}    # form -> list of rule names (L2只警示不排除)

    # L1 硬排除：仅 L1 的排除类动作将概率置零
    for r in all_matched.get('L1', []):
        if r['action'] in ('exclude',):
            for f in r['target_forms']:
                if f in le.classes_:
                    probs[list(le.classes_).index(f)] = 0.0

    # L2 风险警示：不置零，仅记录警告信息（在推理中提示风险）
    for r in all_matched.get('L2', []):
        for f in r['target_forms']:
            if f in le.classes_:
                if f not in l2_warned_forms:
                    l2_warned_forms[f] = []
                l2_warned_forms[f].append({
                    'rule_name': r['rule_name'],
                    'recommendation': r['recommendation'],
                    'source': r['source'],
                })

    # 注意：L3/L4 规则不再干预概率，仅作为解释性证据传入推理层。
    # CatBoost 模型已包含 41 维 KG 特征，概率排序由模型独立完成，
    # 规则匹配结果（all_matched）在推理生成阶段提供结构化论据。
    # 收集 L3/L4 支持的剂型（用于前端展示规则证据，不改变概率）
    for level in ['L3', 'L4']:
        for r in all_matched.get(level, []):
            for f in r.get('target_forms', []):
                if f in le.classes_:
                    rule_evidence.add(f)

    # 重新归一化
    total = probs.sum()
    if total > 0:
        probs = probs / total

    # 重新排序
    top_idx = np.argsort(probs)[::-1][:5]

    top5 = []
    for rank, idx in enumerate(top_idx, 1):
        form = le.classes_[idx]
        top5.append({
            'rank': rank,
            'form': form,
            'confidence': float(probs[idx]),
            'confidence_pct': f'{probs[idx] * 100:.1f}%',
            'rule_evidence': form in rule_evidence,
        })

    # KG 锚定检查：输入处方的药材集合与 KG 中某处方完全一致时，直接锚定
    input_herbs_set = set(normalise_herb(h) for h in herbs)
    anchor_match = None

    for fid, f_herbs in formula_to_herbs.items():
        if input_herbs_set == f_herbs:
            fform = formula_to_form.get(fid)
            fname = formula_to_name.get(fid, '未知方')
            if fform and fform != top5[0]['form']:
                anchor_match = {
                    'name': fname,
                    'form': fform,
                    'overlap': list(input_herbs_set),
                    'overlap_count': len(input_herbs_set)
                }
                # 将锚定剂型置顶
                for i, item in enumerate(top5):
                    if item['form'] == fform:
                        top5.pop(i)
                        break
                top5.insert(0, {
                    'rank': '*',
                    'form': fform,
                    'confidence': 1.0,
                    'confidence_pct': '锚定',
                    'is_anchor': True
                })
                break

    # 规则引擎：基于已计算的 all_matched 生成 excluded 和 warned 列表
    # L1 → excluded（硬约束排除）
    # L2 → warned（风险警示，不排除但需在推理中标注）
    excluded = []
    warned = []
    seen_excluded = {}
    seen_warned = {}
    for r in all_matched.get('L1', []):
        if r['action'] in ('exclude',):
            for f in r['target_forms']:
                if f not in seen_excluded or r['evidence_level'] < seen_excluded[f]:
                    seen_excluded[f] = r['evidence_level']
                    excluded = [e for e in excluded if e['form'] != f]
                    excluded.append({
                        'form': f,
                        'reason': r['recommendation'],
                        'evidence_level': r['evidence_level'],
                        'rule_name': r['rule_name'],
                        'source': r['source'],
                        'action': r['action'],
                    })
    for r in all_matched.get('L2', []):
        for f in r['target_forms']:
            if f not in seen_warned or r['evidence_level'] < seen_warned[f]:
                seen_warned[f] = r['evidence_level']
                warned = [w for w in warned if w['form'] != f]
                warned.append({
                    'form': f,
                    'reason': r['recommendation'],
                    'evidence_level': r['evidence_level'],
                    'rule_name': r['rule_name'],
                    'source': r['source'],
                    'action': r['action'],
                })
    # L2 中没有 target_forms 但仍然有 warn 动作的规则（全局警示）
    for r in all_matched.get('L2', []):
        if not r.get('target_forms') and r['action'] == 'warn':
            warned.append({
                'form': '全部剂型',
                'reason': r['recommendation'],
                'evidence_level': r['evidence_level'],
                'rule_name': r['rule_name'],
                'source': r['source'],
                'action': r['action'],
            })

    # 规则冲突检测：L3/L4 加权方向相反时标注冲突
    rule_conflicts = _detect_rule_conflicts(all_matched)

    # 若 KG 锚定命中，移除锚定剂型从排除列表（已知方知识优先于通用规则）
    if anchor_match:
        excluded = [e for e in excluded if e['form'] != anchor_match['form']]
        warned = [w for w in warned if w['form'] != anchor_match['form']]

    # KG 证据
    evidence = retrieve_kg_evidence(herbs, indication)

    # 推理文本
    reasoning = generate_reasoning(top5, evidence, excluded, herb_props_detail, all_matched, anchor_match)

    return {
        'top5': top5,
        'excluded': excluded,
        'warned': warned,
        'l2_warned_forms': l2_warned_forms,
        'rule_conflicts': rule_conflicts,
        'evidence': {
            'herb_props': evidence['herb_props'],
            'similar_formulas': evidence['similar_formulas'],
            'disease_preference': evidence['disease_preference'],
            'rules': evidence['rules'],
        },
        'reasoning': reasoning,
        'anchor_match': anchor_match,
        'matched_rules': all_matched,
    }


# ==============================================
# Flask 路由
# ==============================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    prescription = data.get('prescription', '').strip()
    indication = data.get('indication', '').strip()
    use_llm = bool(data.get('use_llm', False))

    if not prescription:
        return jsonify({'error': '请输入药材处方'}), 400

    herbs, doses = parse_prescription(prescription)
    if not herbs:
        return jsonify({'error': '未能解析到药材，请使用"药名 克数"格式，如"丹参 15g, 三七 9g"'}), 400

    # 先提取药材属性（用于规则和应用）
    _, herb_props_detail = extract_features(herbs, doses, indication)

    result = predict(herbs, doses, indication, herb_props_detail)
    if result is None:
        return jsonify({'error': '特征提取失败'}), 500

    # 可选：LLM 增强推理（体现 KG-RAR 的 Generate 环节）
    llm_used = False
    llm_failed = False
    if use_llm:
        llm_reasoning, llm_excluded, llm_conflicts = llm_generate_reasoning(
            result['top5'], result['evidence'], herbs, indication, result['matched_rules'])
        if llm_reasoning:
            result['reasoning'] = llm_reasoning
            llm_used = True
        if llm_excluded is not None:
            result['excluded'] = llm_excluded
        if llm_conflicts:
            result['llm_conflicts'] = llm_conflicts
        if not llm_reasoning:
            llm_failed = True
    result['use_llm'] = llm_used
    result['llm_failed'] = llm_failed

    # 统计药材属性
    props_summary = {
        'total': len(herb_props_detail),
        'known': sum(1 for p in herb_props_detail if p.get('found')),
        'has_volatile': any(p.get('volatile') for p in herb_props_detail if p.get('found')),
        'has_toxic': any(p.get('toxic') for p in herb_props_detail if p.get('found')),
        'has_mineral': any(p.get('mineral') for p in herb_props_detail if p.get('found')),
        'has_precious': any(p.get('precious') for p in herb_props_detail if p.get('found')),
        'herbs': herb_props_detail,
    }

    return jsonify({
        'herbs': herbs,
        'doses': doses,
        'indication': indication,
        'props_summary': props_summary,
        'matched_rules': result.get('matched_rules', {}),
        'warned': result.get('warned', []),
        'rule_conflicts': result.get('rule_conflicts', []),
        **result
    })


@app.route('/api/examples')
def api_examples():
    return jsonify([
        {
            'name': '复方丹参片',
            'prescription': '丹参 15g, 三七 9g, 冰片 0.5g',
            'indication': '冠心病心绞痛，气滞血瘀证'
        },
        {
            'name': '一捻金',
            'prescription': '大黄 10g, 牵牛子 10g, 槟榔 10g, 人参 10g, 朱砂 5g',
            'indication': '脾胃不和，痰食阻滞，积滞便秘'
        },
        {
            'name': '麻黄汤',
            'prescription': '麻黄 9g, 桂枝 6g, 苦杏仁 9g, 甘草 6g',
            'indication': '外感风寒，恶寒发热，喘咳'
        },
        {
            'name': '白虎汤',
            'prescription': '石膏 30g, 知母 9g, 甘草 6g, 粳米 15g',
            'indication': '气分热盛，壮热面赤，烦渴引饮'
        },
        {
            'name': '四君子汤',
            'prescription': '人参 9g, 白术 9g, 茯苓 9g, 甘草 6g',
            'indication': '脾胃气虚，面色萎黄，气短乏力'
        },
    ])


# ==============================================
# 启动
# ==============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('\n' + '=' * 60)
    print('  KG-RAR 中药剂型智能推荐系统')
    print(f'  http://localhost:{port}')
    print('=' * 60 + '\n')
    app.run(host='0.0.0.0', port=port, debug=False)