# -*- coding: utf-8 -*-
"""中药处方特征提取器——从药材+适应症+克数→68维手工特征+41维KG特征"""

import re, json, numpy as np
from collections import Counter

# 单源化（2026-10-12）：药材名标准化唯一实现在 src/features/herb_normalize.py，
# 本文件 re-export normalise_herb / ALIAS / EXTRACT_MAP 以兼容旧导入方。
from src.features.herb_normalize import normalise_herb, ALIAS, EXTRACT_MAP

# （原 EXTRACT_MAP / ALIAS / _PROCESS_PREFIXES / _KNOWN_HERBS 已迁移至 src/features/herb_normalize.py）

# 剂型列表（口服液已并入合剂，20→19）
ALL_FORMS = ['丸剂', '片剂', '胶囊剂', '颗粒剂', '合剂', '散剂', '膏剂', '糖浆剂',
             '注射剂', '酒剂', '酊剂', '栓剂', '茶剂', '丹剂', '搽剂', '洗剂',
             '气雾剂', '露剂', '膜剂']

# （原 _PROCESS_PREFIXES / _KNOWN_HERBS 已迁移至 src/features/herb_normalize.py）

# 炮制标记 → 特征类别（用于生成炮制相关特征）
_PROCESS_CATEGORIES = {
    '煅': 'calcined',
    '酒': 'wine_processed', '酒炒': 'wine_processed', '酒炙': 'wine_processed',
    '酒制': 'wine_processed', '酒蒸': 'wine_processed', '酒炖': 'wine_processed',
    '醋': 'vinegar_processed', '醋炒': 'vinegar_processed', '醋炙': 'vinegar_processed',
    '醋制': 'vinegar_processed', '醋蒸': 'vinegar_processed',
    '蜜': 'honey_processed', '蜜炒': 'honey_processed', '蜜炙': 'honey_processed',
    '蜜制': 'honey_processed',
    '盐': 'salt_processed', '盐炒': 'salt_processed', '盐炙': 'salt_processed',
    '盐制': 'salt_processed',
    '姜': 'ginger_processed', '姜汁': 'ginger_processed', '姜炙': 'ginger_processed',
    '姜制': 'ginger_processed', '姜汁炒': 'ginger_processed', '姜汁炙': 'ginger_processed',
    '炒': 'dry_fried', '麸炒': 'dry_fried', '米炒': 'dry_fried', '土炒': 'dry_fried',
    '砂炒': 'dry_fried', '蛤粉炒': 'dry_fried',
    '炙': 'liquid_fried',
    '焦': 'charred',
    '制': 'processed',
    '麸': 'bran_fried',
    '蒸': 'steamed', '煮': 'boiled', '煨': 'simmered',
}


def get_process_features(herbs):
    """从药材列表提取炮制特征（7维）

    通过比较原始输入与标准化结果，检测每味药是否经过炮制，
    并归类到炮制大类。

    Returns:
        dict with keys:
        - has_processed (int): 处方中是否有炮制品
        - processed_ratio (float): 炮制品占比
        - has_calcined (int): 有煅制药材（质地酥脆→散剂友好）
        - has_wine_processed (int): 有酒制药材（缓和寒性，增强活血）
        - has_vinegar_processed (int): 有醋制药材（增强止痛，引药入肝）
        - has_honey_processed (int): 有蜜炙药材（润肺止咳，缓和药性）
        - has_dry_fried (int): 有炒制药材（减毒增效，增强香气）
    """
    n = len(herbs)
    feats = {
        'has_processed': 0,
        'processed_ratio': 0.0,
        'has_calcined': 0,
        'has_wine_processed': 0,
        'has_vinegar_processed': 0,
        'has_honey_processed': 0,
        'has_dry_fried': 0,
    }
    if n == 0:
        return feats

    processed_count = 0
    for herb in herbs:
        original = str(herb).strip()
        normalized = normalise_herb(original)

        # 原始名与标准化名不同 → 可能经过了炮制
        if original == normalized:
            continue

        processed_count += 1
        tags = set()
        for pf, cat in _PROCESS_CATEGORIES.items():
            if pf in original:
                tags.add(cat)

        if 'calcined' in tags:          feats['has_calcined'] = 1
        if 'wine_processed' in tags:    feats['has_wine_processed'] = 1
        if 'vinegar_processed' in tags: feats['has_vinegar_processed'] = 1
        if 'honey_processed' in tags:   feats['has_honey_processed'] = 1
        if tags & {'dry_fried', 'liquid_fried', 'charred', 'bran_fried', 'processed', 'steamed', 'boiled', 'simmered'}:
            feats['has_dry_fried'] = 1  # 广义炒/炙/炮制

    feats['has_processed'] = 1 if processed_count > 0 else 0
    feats['processed_ratio'] = processed_count / n
    return feats


# （normalise_herb 已收敛至 src/features/herb_normalize.py，本文件顶部 re-export）


def parse_prescription(text):
    """解析 "丹参 15g, 三七 9g" 格式的输入"""
    herbs, doses = [], []
    for m in re.findall(r'([一-龥]{1,6}(?:[（(][^）)]+[）)])?)\s*(\d+\.?\d*)\s*(?:g|ml|mg|克|毫升|毫克)?',
                        str(text)):
        name = m[0].strip()
        if len(name) >= 2:
            herbs.append(name)
            doses.append(float(m[1]))
    return herbs, doses


class FeatureExtractor:
    """中药处方特征提取器"""

    def __init__(self, herb_dict, kg):
        """
        Parameters:
        - herb_dict: 药材属性字典 {name: {nature, taste, meridian, ...}}
        - kg: 知识图谱 {'entities': {...}, 'triples': [...]}
        """
        self.herb_dict = herb_dict
        self._build_kg_indexes(kg)

    def _build_kg_indexes(self, kg):
        """构建 KG 索引"""
        id2ent = {int(eid): info['name'] for eid, info in kg['entities'].items()}
        herb_to_formulas = {}
        formula_to_form = {}

        for t in kg['triples']:
            if t['relation_name'] == 'contained_in':
                herb_to_formulas.setdefault(t['head'], []).append(t['tail'])
            elif t['relation_name'] == 'has_form':
                formula_to_form[t['head']] = t['tail_name'].replace('dosage_form:', '')

        # 药材→剂型偏好
        herb_name_to_forms = {}
        for h_id, f_ids in herb_to_formulas.items():
            h_name = id2ent.get(h_id, str(h_id))
            fc = Counter()
            for f_id in f_ids:
                f = formula_to_form.get(f_id)
                if f:
                    fc[f] += 1
            if fc:
                t = sum(fc.values())
                herb_name_to_forms[h_name] = {k: v / t for k, v in fc.most_common()}

        # 适应症→剂型
        ind_to_forms = Counter()
        for t in kg['triples']:
            if t['relation_name'] == 'treats':
                f = formula_to_form.get(t['head'])
                if f:
                    ind_to_forms[(t['tail_name'], f)] += 1

        self.herb_name_to_forms = herb_name_to_forms
        self.ind_to_forms = ind_to_forms

    def extract_features(self, herbs, doses, indication, administration=''):
        """核心特征提取：68维手工特征 + 41维KG特征"""
        n = len(herbs)
        if n == 0:
            return None, []

        feats = {'num_herbs': n}
        # 四气五味折叠：微寒/大寒→寒，微温→温，大热→热；微苦→苦，微甘→甘，微辛→辛
        nature_fold = {'微寒': '寒', '大寒': '寒', '微温': '温', '大热': '热'}
        taste_fold = {'微苦': '苦', '微甘': '甘', '微辛': '辛'}
        nc = {'寒': 0, '凉': 0, '平': 0, '温': 0, '热': 0}
        tc = {'辛': 0, '甘': 0, '酸': 0, '苦': 0, '咸': 0, '淡': 0, '涩': 0}
        mc = {'心': 0, '肝': 0, '脾': 0, '肺': 0, '肾': 0, '胃': 0,
              '胆': 0, '小肠': 0, '大肠': 0, '膀胱': 0, '三焦': 0, '心包': 0}
        xc = {'hard': 0, 'starchy': 0, 'fibrous': 0, 'mucilaginous': 0,
              'oily': 0, 'gelatinous': 0, 'normal': 0}
        cv = ct = cm = cp = kn = 0
        herb_props_detail = []

        for herb in herbs:
            hc = normalise_herb(herb)
            props = self.herb_dict.get(hc) or self.herb_dict.get(herb)  # 优先标准化名
            if props is None:
                # LLM 兜底机制：字典中查不到时，调用 LLM 推断属性
                from src.features.llm_herb_property import infer_herb_properties
                from src.features.herb_category import usable_props
                raw = infer_herb_properties(herb)
                parent_name = raw.get('parent') if isinstance(raw, dict) else None
                props = usable_props(raw, self.herb_dict)  # 分流+缺失标记：非中药/无母药/属性缺失 → 未知
                if props is None:
                    xc['normal'] += 1
                    herb_props_detail.append({'name': herb, 'found': False})
                    continue
                # 有效兜底（herb 或 母药药典值）
                herb_props_detail.append({
                    'name': herb, 'found': True, 'nature': props.get('nature', ''),
                    'taste': props.get('taste', []), 'meridian': props.get('meridian', []),
                    'volatile': props.get('volatile', False), 'toxic': props.get('toxic', False),
                    'mineral': props.get('mineral', False), 'precious': props.get('precious', False),
                    'texture': props.get('texture', 'normal'),
                    '_source': 'dict:parent' if (parent_name and parent_name in self.herb_dict) else 'llm',
                })
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
            herb_props_detail.append({
                'name': herb, 'found': True, 'nature': props.get('nature', ''),
                'taste': props.get('taste', []), 'meridian': props.get('meridian', []),
                'volatile': props.get('volatile', False), 'toxic': props.get('toxic', False),
                'mineral': props.get('mineral', False), 'precious': props.get('precious', False),
                'texture': props.get('texture', 'normal'),
            })

        # —— 手工特征（68维）——
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
        admin = str(administration) if administration else ''
        feats['is_topical'] = int(any(w in admin for w in ['外用','涂','敷','贴','搽','洗','熏','点眼','滴眼','滴鼻','滴耳','灌肠','注入','塞入','塞肛','阴道','吹','喷','滴入']))

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

        # 剂量特征
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

        # —— KG 特征（41维）——
        form_scores = {f: 0.0 for f in ALL_FORMS}
        hc = 0
        for herb in herbs:
            for kg_key in [herb, 'herb:' + herb]:
                if kg_key in self.herb_name_to_forms:
                    for form, prob in self.herb_name_to_forms[kg_key].items():
                        if form in form_scores:
                            form_scores[form] += prob
                    hc += 1
                    break
        if hc > 0:
            for f in ALL_FORMS:
                feats['kg_herb_form_' + f] = form_scores[f] / hc
        else:
            for f in ALL_FORMS:
                feats['kg_herb_form_' + f] = 0.0

        ind_kw = re.findall(r'[一-龥]{2,6}', str(indication) if indication else '')
        ind_fs = Counter()
        for kw in ind_kw[:10]:
            for f in ALL_FORMS:
                ind_fs[f] += self.ind_to_forms.get((kw, f), 0) or self.ind_to_forms.get(('indication:' + kw, f), 0)
        total = sum(ind_fs.values())
        for f in ALL_FORMS:
            feats['kg_ind_form_' + f] = ind_fs[f] / max(total, 1)

        hs = [feats['kg_herb_form_' + f] for f in ALL_FORMS]
        feats['kg_herb_form_max'] = max(hs)
        feats['kg_herb_form_entropy'] = -sum(p * np.log(p + 1e-9) for p in hs if p > 0)
        feats['kg_ind_form_max'] = max([feats['kg_ind_form_' + f] for f in ALL_FORMS])

        return feats, herb_props_detail