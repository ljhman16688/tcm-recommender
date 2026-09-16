# -*- coding: utf-8 -*-
"""
LLM 药材属性推断兜底机制
当药材在 herb_property_dict.json 中查询不到时，调用 LLM 推断性味归经和理化属性，
并将结果缓存到本地 JSON 文件，避免重复调用。
"""

import json
import os
import re

# LLM 配置（从环境变量读取）
LLM_API_KEY = os.environ.get('LLM_API_KEY', '')
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.zhizengzeng.com/v1')
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen3.7-max')

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'data', 'llm_herb_cache.json')

# 允许的枚举值
VALID_NATURES = {'寒', '凉', '平', '温', '热', '微寒', '微温', '大热', '大寒'}
VALID_TASTES = {'辛', '甘', '酸', '苦', '咸', '淡', '涩'}
VALID_MERIDIANS = {'心', '肝', '脾', '肺', '肾', '胃', '胆', '小肠', '大肠', '膀胱', '三焦', '心包'}
VALID_TEXTURES = {'hard', 'starchy', 'fibrous', 'mucilaginous', 'oily', 'gelatinous', 'normal'}


def _load_cache():
    """加载缓存文件"""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    """保存缓存文件"""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _validate_and_fix(herb_name, props):
    """校验并修正 LLM 返回的属性值，确保在合法枚举范围内"""
    # 校验性
    nature = props.get('nature', '平')
    if nature not in VALID_NATURES:
        # 模糊匹配
        for vn in VALID_NATURES:
            if vn in nature:
                nature = vn
                break
        else:
            nature = '平'  # 默认平性

    # 校验味
    taste = props.get('taste', [])
    if isinstance(taste, str):
        taste = [t.strip() for t in taste.replace('、', ',').split(',')]
    taste = [t for t in taste if t in VALID_TASTES]
    if not taste:
        taste = ['甘']  # 默认甘味

    # 校验归经
    meridian = props.get('meridian', [])
    if isinstance(meridian, str):
        meridian = [m.strip() for m in meridian.replace('、', ',').split(',')]
    meridian = [m for m in meridian if m in VALID_MERIDIANS]
    if not meridian:
        meridian = ['肝', '脾']  # 默认归经

    # 校验质地
    texture = props.get('texture', 'normal')
    if texture not in VALID_TEXTURES:
        texture = 'normal'

    return {
        'nature': nature,
        'taste': taste,
        'meridian': meridian,
        'volatile': bool(props.get('volatile', False)),
        'mineral': bool(props.get('mineral', False)),
        'precious': bool(props.get('precious', False)),
        'toxic': bool(props.get('toxic', False)),
        'texture': texture,
        '_source': 'llm',  # 标记来源
    }


def infer_herb_properties(herb_name):
    """
    推断药材属性。优先查缓存，缓存未命中则调用 LLM。
    返回 dict 或 None（LLM 调用失败时）
    """
    cache = _load_cache()

    # 命中缓存
    if herb_name in cache:
        return cache[herb_name]

    # 调用 LLM
    print(f'  [LLM] 推断药材属性: {herb_name} ...')
    props = _call_llm(herb_name)

    if props is None:
        return None

    # 校验并缓存
    validated = _validate_and_fix(herb_name, props)
    cache[herb_name] = validated
    _save_cache(cache)

    print(f'  [LLM] {herb_name} → 性:{validated["nature"]} 味:{"/".join(validated["taste"])} '
          f'归经:{"/".join(validated["meridian"])} 挥发:{validated["volatile"]} '
          f'毒性:{validated["toxic"]} 质地:{validated["texture"]}')

    return validated


def _call_llm(herb_name):
    """调用 LLM 获取药材属性"""
    prompt = f"""你是资深中药学专家。请根据中药学知识，给出以下药材的属性信息。只需返回 JSON，不要任何解释。

药材名：{herb_name}

请返回如下格式的 JSON：
{{
    "nature": "寒/凉/平/温/热/微寒/微温/大热/大寒 之一",
    "taste": ["辛/甘/酸/苦/咸/淡/涩 中选择"],
    "meridian": ["心/肝/脾/肺/肾/胃/胆/小肠/大肠/膀胱/三焦/心包 中选择"],
    "volatile": true/false,
    "mineral": true/false,
    "precious": true/false,
    "toxic": true/false,
    "texture": "hard/starchy/fibrous/mucilaginous/oily/gelatinous/normal 之一"
}}

判断标准：
- volatile（挥发性）：含挥发油、芳香性成分的药材为 true（如薄荷、砂仁、川芎、当归、肉桂等）
- mineral（矿物药）：矿物类药材为 true（如石膏、朱砂、龙骨等）
- precious（贵重药）：贵重药材为 true（如人参、鹿茸、麝香、牛黄等）
- toxic（毒性）：有毒或小毒的药材为 true（如附子、半夏、细辛等）
- texture（质地）：starchy=富含淀粉（如山药、茯苓），fibrous=纤维性强（如甘草、黄芪），mucilaginous=黏腻（如熟地黄、阿胶），oily=油性（如杏仁、桃仁），hard=坚硬（如矿物），gelatinous=胶质（如龟甲胶），normal=一般"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            max_tokens=500,
        )
        content = resp.choices[0].message.content or ''

        # 提取 JSON
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            return json.loads(m.group())
        return None
    except Exception as e:
        print(f'  [LLM] 推断失败: {e}')
        return None


def get_cache_stats():
    """获取缓存统计"""
    cache = _load_cache()
    return {
        'cached_count': len(cache),
        'cached_herbs': list(cache.keys()),
    }