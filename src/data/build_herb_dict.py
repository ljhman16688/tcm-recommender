# -*- coding: utf-8 -*-
"""
Build herb property dictionary from Sheet3 + manual rules + textbook.
Output: data/herb_property_dict.json (747 herbs)
"""

import pandas as pd
import json
import re
from collections import Counter


def parse_nature_taste_meridian(text):
    """Parse '辛、苦，凉；有小毒。归肺、肝经。' into dict"""
    if pd.isna(text):
        return {'taste': [], 'nature': '', 'toxic': False, 'meridian': []}
    text = str(text).strip()
    result = {'taste': [], 'nature': '', 'toxic': False, 'meridian': []}

    if '归' in text:
        before_gui, after_gui = text.split('归', 1)
    else:
        before_gui, after_gui = text, ''

    before_gui = before_gui.rstrip('。，；')

    if '小毒' in before_gui or '有毒' in before_gui or '大毒' in before_gui:
        result['toxic'] = True
        before_gui = re.sub(r'[；;]\s*有(?:小|大)?毒', '', before_gui)

    parts = re.split(r'[，,]', before_gui)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        nature_part = parts[-1]
        nature_kw = ['大热', '大寒', '微寒', '微温', '寒', '热', '温', '凉', '平']
        for kw in sorted(nature_kw, key=lambda x: -len(x)):
            if kw in nature_part:
                result['nature'] = kw
                break
        taste_set = set()
        for tp in parts[:-1]:
            for t in re.split(r'[、/]', tp):
                t = t.strip()
                if t in ['辛', '甘', '酸', '苦', '咸', '淡', '涩', '微苦', '微甘']:
                    taste_set.add(t)
        result['taste'] = sorted(taste_set)
    elif len(parts) == 1:
        p = parts[0]
        for t in ['辛', '甘', '酸', '苦', '咸', '淡', '涩']:
            if t in p: result['taste'].append(t)
        for n in ['大热', '大寒', '微寒', '微温', '寒', '热', '温', '凉', '平']:
            if n in p: result['nature'] = n; break

    after_gui = after_gui.strip().rstrip('。；， ').replace('经', '')
    meridian_kw = ['心', '肝', '脾', '肺', '肾', '胃', '胆', '小肠', '大肠', '膀胱', '三焦', '心包']
    for m in re.split(r'[、，,]', after_gui):
        m = m.strip()
        if m in meridian_kw:
            result['meridian'].append(m)

    return result


# ===== Volatile herbs (aromatic, essential oil rich) =====
VOLATILE_HERBS = {
    '麻黄','桂枝','紫苏','紫苏叶','紫苏梗','生姜','香薷',
    '荆芥','防风','羌活','白芷','细辛','藁本','苍耳子','辛夷',
    '薄荷','牛蒡子','蝉蜕','桑叶','菊花','蔓荆子','柴胡','升麻','葛根',
    '广藿香','藿香','佩兰','苍术','厚朴','砂仁','白豆蔻','豆蔻','草果',
    '附子','肉桂','桂皮','干姜','高良姜','花椒','小茴香','八角茴香',
    '丁香','荜茇','胡椒','山柰','山奈',
    '陈皮','化橘红','橘红','青皮','枳实','枳壳','木香','香附',
    '乌药','沉香','檀香','川楝子','薤白','大腹皮','甘松','佛手','香橼',
    '麝香','冰片','苏合香','安息香','石菖蒲',
    '樟脑','乳香','没药','川芎','当归','莪术','姜黄','郁金',
}

# ===== Mineral/shell herbs =====
MINERAL_HERBS = {
    '石膏','寒水石','滑石','芒硝','磁石','赭石','自然铜','赤石脂','炉甘石',
    '雄黄','轻粉','龙骨','龙齿','钟乳石','金礞石','青礞石',
    '朱砂','硫磺','硼砂','明矾','枯矾',
    '牡蛎','石决明','珍珠母','珍珠','海蛤壳','瓦楞子',
    '龟甲','鳖甲','琥珀',
}

# ===== Precious herbs =====
PRECIOUS_HERBS = {
    '麝香','牛黄','人工牛黄','人参','西洋参','鹿茸','鹿角','羚羊角',
    '冬虫夏草','三七','血竭','穿山甲','熊胆','蟾酥',
    '藏红花','西红花','海马','海龙','珍珠','沉香','檀香',
    '阿胶','龟甲胶','鹿角胶',
}

# ===== Texture classification =====
TEXTURE_MAP = {
    'hard': {'石膏','龙骨','龙齿','牡蛎','石决明','磁石','赭石','龟甲','鳖甲',
             '穿山甲','鹿角','羚羊角','沉香','檀香','降香','苏木','血竭','琥珀',
             '三棱','莪术','山豆根','金礞石','青礞石','钟乳石','自然铜',
             '寒水石','滑石','赤石脂','炉甘石','海蛤壳','瓦楞子','珍珠母','珍珠',
             '三七','浙贝母','延胡索','郁金'},
    'starchy': {'山药','茯苓','天花粉','葛根','薏苡仁','白芷','白及','半夏',
                '天南星','浙贝母','川贝母','芡实','莲子','白扁豆','防己','赤小豆',
                '土茯苓','何首乌','白芍','板蓝根','白术','苍术'},
    'fibrous': {'麻黄','桂枝','桑白皮','枇杷叶','大腹皮','厚朴','杜仲','黄柏',
                '肉桂','桂皮','竹茹','益母草','桑寄生','钩藤','鸡血藤',
                '淫羊藿','巴戟天','续断','秦艽','威灵仙','荆芥','薄荷',
                '紫苏','广藿香','藿香','佩兰','艾叶','桑叶','荷叶'},
    'mucilaginous': {'知母','玉竹','黄精','麦冬','天冬','玄参','生地黄','熟地黄',
                     '枸杞子','石斛','百合','北沙参','南沙参','芦根','白茅根',
                     '车前草','车前子','胖大海','罗汉果','白及'},
    'oily': {'桃仁','苦杏仁','柏子仁','火麻仁','郁李仁','黑芝麻','核桃仁',
             '苏子','莱菔子','酸枣仁','巴豆','使君子'},
    'gelatinous': {'阿胶','鹿角胶','龟甲胶','黄明胶'},
}

TEXTURE_REVERSE = {}
for tex_type, herb_set in TEXTURE_MAP.items():
    for herb in herb_set:
        TEXTURE_REVERSE[herb] = tex_type


# ===== Supplement herbs not in Sheet3 =====
SUPPLEMENT_HERBS = {
    '冰片': {'nature': '微寒', 'taste': ['辛','苦'], 'meridian': ['心','脾','肺'],
             'volatile': True, 'toxic': False, 'mineral': False, 'precious': False, 'texture': 'normal'},
    '延胡索': {'nature': '温', 'taste': ['辛','苦'], 'meridian': ['肝','脾'],
              'volatile': False, 'toxic': False, 'mineral': False, 'precious': False, 'texture': 'hard'},
    '土鳖虫': {'nature': '寒', 'taste': ['咸'], 'meridian': ['肝'],
              'volatile': False, 'toxic': True, 'mineral': False, 'precious': False, 'texture': 'normal'},
    '紫河车': {'nature': '温', 'taste': ['甘','咸'], 'meridian': ['肺','肝','肾'],
              'volatile': False, 'toxic': False, 'mineral': False, 'precious': True, 'texture': 'normal'},
    '关木通': {'nature': '寒', 'taste': ['苦'], 'meridian': ['心','小肠','膀胱'],
              'volatile': False, 'toxic': True, 'mineral': False, 'precious': False, 'texture': 'fibrous'},
    '白花蛇舌草': {'nature': '寒', 'taste': ['微苦','甘'], 'meridian': ['胃','大肠','小肠'],
                  'volatile': False, 'toxic': False, 'mineral': False, 'precious': False, 'texture': 'fibrous'},
    '六神曲': {'nature': '温', 'taste': ['甘','辛'], 'meridian': ['脾','胃'],
              'volatile': False, 'toxic': False, 'mineral': False, 'precious': False, 'texture': 'starchy'},
}


def normalize_herb_name(name):
    """Remove processing markers from herb names"""
    if not name or pd.isna(name): return ''
    name = str(name).strip()
    name = re.sub(r'[（(](?:酒制|醋制|盐制|姜制|蜜制|炒|煅|蒸|制|熟|焦|炭|麸炒|砂炒|醋炙|酒炙|盐炙|去心|去壳|去皮|去毛)[）)]', '', name)
    name = re.sub(r'[（(][^）)]*?[）)]', '', name)
    name = re.sub(r'^(?:生|鲜|炒|酒|醋|盐|姜|蜜|焦|煅|制|熟|土炒|麸炒|蒸|炙)', '', name)
    alias_map = {
        '萸肉': '山茱萸', '酒萸肉': '山茱萸', '法半夏': '半夏', '清半夏': '半夏',
        '姜半夏': '半夏', '胆南星': '天南星', '炙甘草': '甘草', '炙黄芪': '黄芪',
        '焦山楂': '山楂', '焦麦芽': '麦芽', '焦槟榔': '槟榔', '焦栀子': '栀子',
        '煅龙骨': '龙骨', '煅牡蛎': '牡蛎', '煅石膏': '石膏',
        '熟地黄': '地黄', '生地黄': '地黄', '醋柴胡': '柴胡', '醋香附': '香附',
        '醋延胡索': '延胡索', '制川乌': '川乌', '制草乌': '草乌', '制何首乌': '何首乌',
        '制天南星': '天南星', '煨肉豆蔻': '肉豆蔻',
    }
    return alias_map.get(name, name)


def build_herb_property_dict():
    """Build complete herb property dict from Sheet3 + manual rules"""
    df3 = pd.read_excel('data/others.xlsx', sheet_name='Sheet3', engine='openpyxl')
    df3.columns = ['herb_name', 'assay', 'nature_taste_meridian', 'indication',
                   'dosage', 'precautions', 'col6', 'col7']

    herb_dict = {}

    # Parse Sheet3
    for _, row in df3.iterrows():
        name = str(row['herb_name']).strip()
        if not name or name == 'nan': continue
        props = parse_nature_taste_meridian(row['nature_taste_meridian'])
        herb_dict[name] = {
            'nature': props['nature'],
            'taste': props['taste'],
            'meridian': props['meridian'],
        }

    # Add manual labels
    for name in herb_dict:
        herb_dict[name]['volatile'] = name in VOLATILE_HERBS
        herb_dict[name]['mineral'] = name in MINERAL_HERBS
        herb_dict[name]['precious'] = name in PRECIOUS_HERBS
        herb_dict[name]['toxic'] = herb_dict[name].get('toxic', False) or name in {
            '马钱子','巴豆','斑蝥','轻粉','砒石','砒霜','雄黄','雌黄','升药','铅丹'}
        herb_dict[name]['texture'] = TEXTURE_REVERSE.get(name, 'normal')

    # Add herbs from manual sets not in Sheet3
    extra = (VOLATILE_HERBS | MINERAL_HERBS | PRECIOUS_HERBS) - set(herb_dict.keys())
    for name in extra:
        herb_dict[name] = {
            'nature': '', 'taste': [], 'meridian': [],
            'volatile': name in VOLATILE_HERBS,
            'mineral': name in MINERAL_HERBS,
            'precious': name in PRECIOUS_HERBS,
            'toxic': name in {'马钱子','巴豆','斑蝥','轻粉','砒石','砒霜','雄黄','雌黄','升药','铅丹'},
            'texture': TEXTURE_REVERSE.get(name, 'normal'),
        }

    # Merge supplements
    for name, props in SUPPLEMENT_HERBS.items():
        if name in herb_dict:
            for k, v in props.items():
                if not herb_dict[name].get(k): herb_dict[name][k] = v
        else:
            herb_dict[name] = {
                'nature': props.get('nature',''), 'taste': props.get('taste',[]),
                'meridian': props.get('meridian',[]), 'volatile': props.get('volatile',False),
                'toxic': props.get('toxic',False), 'mineral': props.get('mineral',False),
                'precious': props.get('precious',False), 'texture': props.get('texture','normal'),
            }

    return herb_dict


def extract_prescription_features(herb_list, herb_dict):
    """Extract aggregated features from a herb list for a prescription"""
    n = len(herb_list)
    features = {'num_herbs': n}

    count_volatile = count_toxic = count_mineral = count_precious = known_count = 0
    nature_counter = Counter()
    taste_counter = Counter()
    meridian_counter = Counter()
    texture_counter = Counter()

    for herb in herb_list:
        herb_clean = normalize_herb_name(herb)
        props = herb_dict.get(herb, herb_dict.get(herb_clean))
        if props is None:
            props = {'nature':'','taste':[],'meridian':[],'volatile':False,'toxic':False,'mineral':False,'precious':False,'texture':'normal'}
        else:
            known_count += 1

        if props.get('volatile'): count_volatile += 1
        if props.get('toxic'): count_toxic += 1
        if props.get('mineral'): count_mineral += 1
        if props.get('precious'): count_precious += 1

        if props.get('nature'): nature_counter[props['nature']] += 1
        for t in props.get('taste', []): taste_counter[t] += 1
        for m in props.get('meridian', []): meridian_counter[m] += 1
        texture_counter[props.get('texture', 'normal')] += 1

    features['has_volatile'] = int(count_volatile > 0)
    features['has_toxic'] = int(count_toxic > 0)
    features['has_mineral'] = int(count_mineral > 0)
    features['has_precious'] = int(count_precious > 0)
    features['volatile_ratio'] = count_volatile / n
    features['toxic_ratio'] = count_toxic / n
    features['mineral_ratio'] = count_mineral / n
    features['precious_ratio'] = count_precious / n
    features['known_ratio'] = known_count / n

    for nat in ['寒','凉','平','温','热']:
        features[f'nature_{nat}_ratio'] = nature_counter.get(nat, 0) / n
    for t in ['辛','甘','酸','苦','咸','淡','涩']:
        features[f'taste_{t}_ratio'] = taste_counter.get(t, 0) / n
    for m in ['心','肝','脾','肺','肾','胃','胆','小肠','大肠','膀胱','三焦','心包']:
        features[f'meridian_{m}_ratio'] = meridian_counter.get(m, 0) / n
    for tex in ['hard','starchy','fibrous','mucilaginous','oily','gelatinous']:
        features[f'texture_{tex}_ratio'] = texture_counter.get(tex, 0) / n

    return features


if __name__ == '__main__':
    herb_dict = build_herb_property_dict()
    print(f'Herb dict: {len(herb_dict)} herbs')
    with open('data/herb_property_dict.json', 'w', encoding='utf-8') as f:
        json.dump(herb_dict, f, ensure_ascii=False, indent=2)
    print('Saved to data/herb_property_dict.json')
