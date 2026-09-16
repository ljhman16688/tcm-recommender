# -*- coding: utf-8 -*-
"""中药处方特征提取器——从药材+适应症+克数→68维手工特征+41维KG特征"""

import re, json, numpy as np
from collections import Counter

# ==============================================
# 药材名标准化映射表
# ==============================================
EXTRACT_MAP = {
    '广藿香油': '广藿香', '紫苏叶油': '紫苏叶', '甘草浸膏': '甘草',
    '薄荷油': '薄荷', '薄荷脑': '薄荷', '薄荷素油': '薄荷',
    '丁香油': '丁香', '丁香罗勒油': '丁香',
    '肉桂油': '肉桂', '桂皮油': '肉桂',
    '刺五加浸膏': '刺五加', '颠茄流浸膏': '颠茄草',
    '当归油': '当归', '川芎油': '川芎',
    '八角茴香油': '八角茴香', '小茴香油': '小茴香',
    '干姜油': '干姜', '生姜油': '生姜',
    '陈皮油': '陈皮', '砂仁叶油': '砂仁',
    '荆芥油': '荆芥', '荆芥穗油': '荆芥',
    '降香油': '降香', '檀香油': '檀香',
    '蛇床子油': '蛇床子', '艾叶油': '艾叶',
    '藿香油': '广藿香', '满山红油': '满山红',
    '水牛角浓缩粉': '水牛角', '水牛角浓缩丸': '水牛角',
    '人工牛黄': '牛黄', '人工麝香': '麝香',
    '羚羊角粉': '羚羊角', '珍珠粉': '珍珠',
    '三七粉': '三七', '熊胆粉': '熊胆',
}

ALIAS = {
    '萸肉': '山茱萸', '法半夏': '半夏', '清半夏': '半夏', '姜半夏': '半夏',
    '胆南星': '胆南星',   # 胆汁制，毒性显著降低；保持独立实体，不与生天南星混淆
    '制首乌': '制何首乌',  # 补肝肾、益精血，安全
    '生首乌': '生何首乌',  # 肝毒性风险，与制何首乌严格区分
    '炙甘草': '甘草',
    '熟地黄': '熟地黄',  # 微温，滋阴补血；不与生地黄（寒性）混淆
    '生地黄': '生地黄',  # 寒性，清热凉血；保持独立
    '鲜地黄': '生地黄',  # 鲜地黄与生地黄药性相近
    '白茯苓': '茯苓', '云茯苓': '茯苓', '赤茯苓': '茯苓',
    '於术': '白术', '于术': '白术', '生白术': '白术', '炒白术': '白术',
    '白芍药': '白芍', '杭白芍': '白芍', '生白芍': '白芍', '炒白芍': '白芍', '酒白芍': '白芍',
    '川芎藭': '川芎', '抚芎': '川芎',
    '当归身': '当归', '当归尾': '当归', '全当归': '当归',
    '淮山药': '山药', '怀山药': '山药', '薯蓣': '山药',
    '川黄连': '黄连', '雅连': '黄连', '炒黄连': '黄连',
    '枯芩': '黄芩', '条芩': '黄芩', '子芩': '黄芩',
    '绵黄芪': '黄芪', '生黄芪': '黄芪', '炙黄芪': '黄芪',
    '官桂': '肉桂', '桂心': '肉桂', '上肉桂': '肉桂',
    '炮附子': '附子', '制附子': '附子', '黑附子': '附子',
    '制半夏': '半夏', '姜半夏': '半夏', '法半夏': '半夏',
    '茅术': '苍术', '制苍术': '苍术',
    '广陈皮': '陈皮', '新会皮': '陈皮', '橘皮': '陈皮', '橘红': '陈皮',
    '淮牛膝': '牛膝', '怀牛膝': '牛膝', '川牛膝': '牛膝',
    '北柴胡': '柴胡', '软柴胡': '柴胡',
    '川大黄': '大黄', '锦纹': '大黄', '生大黄': '大黄', '酒大黄': '大黄',
    '金银花': '金银花', '双花': '金银花', '忍冬': '金银花',
    '瓜蒌': '瓜蒌', '全瓜蒌': '瓜蒌', '瓜蒌皮': '瓜蒌',
    '麦门冬': '麦冬', '天门冬': '天冬',
    '山萸肉': '山茱萸', '山萸': '山茱萸',
    '焦山楂': '山楂', '生山楂': '山楂', '炒山楂': '山楂',
    '建曲': '神曲', '六神曲': '神曲', '炒神曲': '神曲',
    '生牡蛎': '牡蛎', '煅牡蛎': '牡蛎',
    '生龙骨': '龙骨', '煅龙骨': '龙骨',
    '川厚朴': '厚朴', '姜厚朴': '厚朴', '制厚朴': '厚朴',
    '潞党参': '党参', '台党参': '党参', '西党参': '党参',
    '北沙参': '北沙参',  # 伞形科，与南沙参（桔梗科）不同基原
    '南沙参': '南沙参',
    '玄参': '玄参', '元参': '玄参',
    '川贝母': '川贝母', '浙贝母': '浙贝母', '象贝': '浙贝母',
    '白芥子': '白芥子', '黄芥子': '芥子',
    '草决明': '决明子', '石决明': '石决明',
    '抚川芎': '川芎',
    '苏薄荷': '薄荷', '南薄荷': '薄荷',
    '冬桑叶': '桑叶', '霜桑叶': '桑叶',
    '鲜地黄': '生地黄',
    '乌贼骨': '海螵蛸',
    '元胡': '延胡索', '玄胡': '延胡索', '延胡': '延胡索',
    '粉防己': '防己', '汉防己': '防己',
    '栝楼': '瓜蒌', '栝楼根': '天花粉',
    '潞参': '党参', '台乌药': '乌药',
    '仙灵脾': '淫羊藿', '破故纸': '补骨脂',
    '白僵蚕': '僵蚕', '蝉衣': '蝉蜕',
    '地鳖虫': '土鳖虫', '庶虫': '土鳖虫',
    '云木香': '木香', '广木香': '木香',
    '天花粉': '天花粉', '栝蒌根': '天花粉',
    '北五味子': '五味子', '南五味子': '五味子',
    '杭菊花': '菊花', '滁菊花': '菊花', '甘菊花': '菊花',
    '甘枸杞': '枸杞子', '宁枸杞': '枸杞子',
    '怀生地': '地黄', '远志肉': '远志', '制远志': '远志',
    '粉甘草': '甘草', '生甘草': '甘草', '国老': '甘草',
    '大红枣': '大枣', '净麻黄': '麻黄', '川桂枝': '桂枝',
    '生石膏': '石膏', '煅石膏': '石膏',
    '光杏仁': '苦杏仁', '苦杏': '苦杏仁',
    '鲜生姜': '生姜', '干生姜': '干姜',
    '川羌活': '羌活', '西羌活': '羌活',
    '香白芷': '白芷', '杭白芷': '白芷',
    '苦桔梗': '桔梗', '甜桔梗': '桔梗',
    '生薏仁': '薏苡仁', '炒薏仁': '薏苡仁', '苡仁': '薏苡仁',
    '白蔻仁': '豆蔻', '白豆蔻': '豆蔻',
    '缩砂仁': '砂仁', '阳春砂': '砂仁',
    '潞党': '党参', '真阿胶': '阿胶', '驴皮胶': '阿胶',
    '茜草根': '茜草', '侧柏炭': '侧柏叶', '藕节炭': '藕节',
    '小蓟炭': '小蓟', '血余炭': '血余', '陈棕炭': '棕榈',
    '百草霜': '百草霜', '伏龙肝': '伏龙肝',
    '风化硝': '芒硝', '玄明粉': '芒硝', '皮硝': '芒硝',
    '明矾': '白矾', '枯矾': '白矾',
    '朱砂': '朱砂', '辰砂': '朱砂', '丹砂': '朱砂',
    '煅磁石': '磁石', '灵磁石': '磁石',
    '代赭石': '代赭石', '赭石': '代赭石',
    '花蕊石': '花蕊石', '赤石脂': '赤石脂', '禹余粮': '禹余粮',
    '煅瓦楞子': '瓦楞子', '海蛤壳': '海蛤壳',
    '沉香曲': '沉香', '檀香末': '檀香',
}

# 剂型列表
ALL_FORMS = ['丸剂', '片剂', '胶囊剂', '颗粒剂', '口服液', '散剂', '膏剂', '糖浆剂',
             '注射剂', '酒剂', '酊剂', '栓剂', '茶剂', '丹剂', '搽剂', '洗剂',
             '气雾剂', '露剂', '膜剂']

# 炮制标记前缀（长度降序，优先匹配最长）
_PROCESS_PREFIXES = [
    '姜汁炙', '蛤粉炒', '姜汁炒',
    '酒炒', '醋炒', '麸炒', '米炒', '土炒', '砂炒', '盐炒', '蜜炒', '姜汁',
    '酒炙', '醋炙', '蜜炙', '盐炙', '姜炙', '油炙',
    '酒制', '醋制', '姜制', '盐制', '蜜制', '水制', '泔制',
    '酒蒸', '醋蒸', '酒炖', '姜炖',
    '炙', '炒', '酒', '醋', '姜', '盐', '蜜', '煅', '焦',
    '制', '法', '麸', '煨', '蒸', '煮', '淬', '漂', '泡', '浸',
]
# 已知标准药材名集合（ALIAS.keys ∪ ALIAS.values），用于校验剥离后的基原名称
_KNOWN_HERBS = set(ALIAS.keys()) | set(ALIAS.values())

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


def normalise_herb(name):
    """药材名标准化：ALIAS查表 → 炮制前缀剥离 → EXTRACT_MAP"""
    name = str(name).strip()
    name = re.sub(r'[（(][^)）]*[)）]', '', name)

    # 1. 精确ALIAS查表
    name = ALIAS.get(name, name)

    # 2. 若仍非已知药材名，迭代剥离炮制前缀
    if name not in _KNOWN_HERBS:
        changed = True
        while changed:
            changed = False
            for pf in _PROCESS_PREFIXES:
                if name.startswith(pf) and len(name) > len(pf):
                    base = name[len(pf):]
                    # 基原名必须看起来像个药材：≥2汉字，且（在已知集合中 或 为纯汉字）
                    looks_valid = base in _KNOWN_HERBS or bool(re.match(r'^[一-龥]{2,}$', base))
                    if looks_valid:
                        name = ALIAS.get(base, base)
                        changed = True
                        break
                    mapped = ALIAS.get(base)
                    if mapped and mapped in _KNOWN_HERBS:
                        name = mapped
                        changed = True
                        break

    name = EXTRACT_MAP.get(name, name)
    return name


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

    def extract_features(self, herbs, doses, indication):
        """核心特征提取：68维手工特征 + 41维KG特征"""
        n = len(herbs)
        if n == 0:
            return None, []

        feats = {'num_herbs': n}
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
                props = infer_herb_properties(herb)
                if props is None:
                    xc['normal'] += 1
                    herb_props_detail.append({'name': herb, 'found': False})
                    continue
                # 标记为 LLM 推断
                herb_props_detail.append({
                    'name': herb, 'found': True, 'nature': props.get('nature', ''),
                    'taste': props.get('taste', []), 'meridian': props.get('meridian', []),
                    'volatile': props.get('volatile', False), 'toxic': props.get('toxic', False),
                    'mineral': props.get('mineral', False), 'precious': props.get('precious', False),
                    'texture': props.get('texture', 'normal'),
                    '_source': 'llm',
                })
                kn += 1
                if props.get('volatile'): cv += 1
                if props.get('toxic'): ct += 1
                if props.get('mineral'): cm += 1
                if props.get('precious'): cp += 1
                nat = props.get('nature', '')
                if nat in nc: nc[nat] += 1
                for t in props.get('taste', []):
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
            if nat in nc: nc[nat] += 1
            for t in props.get('taste', []):
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
        feats['is_topical'] = int(any(w in ind for w in ['外用', '涂', '敷', '贴']))

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