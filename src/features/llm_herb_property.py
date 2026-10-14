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
    os.path.abspath(__file__)))), 'data', 'herb_llm_cache.json')

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


def _is_valid_herb_name(name):
    """预检：输入是否为合法的中药材名称。
    返回 (is_valid, reason)。
    """
    # 1. 长度检查：药材名通常 2-8 个中文字符（含炮制前缀/后缀）
    if len(name) < 2:
        return False, 'name_too_short'
    if len(name) > 12:
        return False, 'name_too_long'

    # 2. 含剂量/化验/工艺关键词 → 不是药材名
    dosage_assay_kw = [
        '每粒', '每袋', '每片', '每丸', '每味', '每服',
        '含总', '含钙', '含锌', '含铁', '含量',
        '不少于', '不多于', '不得少于', '不得多于',
        '相当于', '折合', '约对应', '约合',
        '总计', '总量', '总量按',
        '标注', '补注', '投料',
        '提取制成', '制成', '收集', '取新鲜',
        '干燥品', '无水物', '按干燥品',
        '浸出物', '测定', '鉴别', '色谱',
    ]
    for kw in dosage_assay_kw:
        if kw in name:
            return False, f'dosage_assay_kw:{kw}'

    # 3. 西药/化学药关键词
    chem_kw = [
        '盐酸', '马来酸', '对乙酰', '咖啡因', '阿司匹林', '布洛芬',
        '氯苯那敏', '苯海拉明', '麻黄碱', '小檗碱', '格列', '沙星',
        '头孢', '霉素', '西林', '替丁', '拉唑', '地平', '他汀',
    ]
    for kw in chem_kw:
        if kw in name:
            return False, f'chemical_kw:{kw}'

    # 4. 辅料/食品关键词
    excipient_kw = [
        '凡士林', '石蜡', '羊毛脂', '甘油', '植物油',
        '蔗糖', '淀粉', '糊精', '明胶', '胶囊壳',
        '香精', '矫味剂', '甜味素',
    ]
    if name in excipient_kw:
        return False, f'excipient:{name}'

    return True, 'ok'




def _validate_and_fix(herb_name, props):
    """校验并修正 LLM 返回的属性值。

    关键变更：如果 LLM 明确标记 is_herb=False，或所有核心字段为空，
    说明 LLM 判定这不是药材 —— 此时不填默认值，返回标记为无效的空结构。
    """
    # --- 先检查 LLM 自己的判断 ---
    is_herb = props.get('is_herb', None)
    # 兼容旧格式：is_herb 可能不存在（旧缓存），此时看核心字段是否全空
    nature_raw = props.get('nature', '')
    taste_raw = props.get('taste', [])
    meridian_raw = props.get('meridian', [])

    if isinstance(taste_raw, str):
        taste_raw = [t.strip() for t in taste_raw.replace('、', ',').split(',') if t.strip()]
    if isinstance(meridian_raw, str):
        meridian_raw = [m.strip() for m in meridian_raw.replace('、', ',').split(',') if m.strip()]

    # 判断 LLM 是否明确返回了"我不知道"信号
    llm_says_not_herb = (
        is_herb is False or
        (is_herb is None and not nature_raw and not taste_raw and not meridian_raw)
    )

    if llm_says_not_herb:
        return {
            'nature': '',
            'taste': [],
            'meridian': [],
            'volatile': False,
            'mineral': False,
            'precious': False,
            'toxic': False,
            'texture': 'normal',
            '_source': 'llm',
            '_invalid': True,  # 标记：非药材，不建议使用
        }

    # --- 校验性 ---
    nature = nature_raw if nature_raw in VALID_NATURES else '平'
    if nature_raw not in VALID_NATURES and nature_raw:
        for vn in VALID_NATURES:
            if vn in nature_raw:
                nature = vn
                break
        else:
            nature = '平'

    # --- 校验味 ---
    taste = [t for t in taste_raw if t in VALID_TASTES]

    # --- 校验归经 ---
    meridian = [m for m in meridian_raw if m in VALID_MERIDIANS]

    # --- 校验质地 ---
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
        '_source': 'llm',
    }


def infer_herb_properties(herb_name):
    """
    推断药材属性。优先查缓存，缓存未命中则调用 LLM。
    返回 dict 或 None（无效输入/LLM 调用失败时）
    """
    cache = _load_cache()

    # 命中缓存
    if herb_name in cache:
        entry = cache[herb_name]
        # 分流 + 缺失标记：无效/非中药/属性缺失的条目一律视为未知，不再当作"已知"
        if entry.get('_invalid') or entry.get('_property_valid') is False:
            return None
        return entry

    # 预检：输入是否合法
    is_valid, reason = _is_valid_herb_name(herb_name)
    if not is_valid:
        print(f'  [LLM] 跳过非药材输入: {herb_name} ({reason})')
        # 缓存为无效条目，避免重复判断
        invalid_entry = {
            'nature': '', 'taste': [], 'meridian': [],
            'volatile': False, 'mineral': False, 'precious': False,
            'toxic': False, 'texture': 'normal',
            '_source': 'llm', '_invalid': True,
        }
        cache[herb_name] = invalid_entry
        _save_cache(cache)
        return None

    if not LLM_API_KEY:
        return None

    # 调用 LLM
    print(f'  [LLM] 推断药材属性: {herb_name} ...')
    props = _call_llm(herb_name)

    if props is None:
        return None

    # 校验并缓存
    validated = _validate_and_fix(herb_name, props)
    cache[herb_name] = validated
    _save_cache(cache)

    if validated.get('_invalid'):
        print(f'  [LLM] {herb_name} → LLM 判定非药材，已跳过')
    else:
        print(f'  [LLM] {herb_name} → 性:{validated["nature"]} 味:{"/".join(validated["taste"])} '
              f'归经:{"/".join(validated["meridian"])} 挥发:{validated["volatile"]} '
              f'毒性:{validated["toxic"]} 质地:{validated["texture"]}')

    return validated


def _call_llm(herb_name):
    """调用 LLM 获取药材属性"""
    prompt = f"""你是资深中药学专家。你的任务是判断给定的名称是否为真实的中药材，并给出属性信息。

【第一步：真实性判断】
- 输入的名称应该是以中文学名、炮制品名、习用名或法定加工品名存在的单味中药材。
- 以下类型 NOT 是药材，必须返回 is_herb: false：
  * 西药、化学药、化合物（如"格列本脲""盐酸小檗碱""咖啡因"）
  * 辅料、食品（如"凡士林""淀粉""花生米"）
  * 剂量说明、化验指标、工艺描述（如"每粒含""不少于""相当于""提取制成"）
  * 明显的文本碎片（如"仅标注""计不得少于"）
  * 中成药、方剂名（多味药组成的复方）
- 如果你不认识这个名称，或无法确定它是单味药材，设置 is_herb: false 并将 nature/taste/meridian 留空数组。

【第二步：属性填写】（仅在 is_herb 为 true 时填写）
- nature：必须从 [寒, 凉, 平, 温, 热, 微寒, 微温, 大热, 大寒] 中选择一个
- taste：从 [辛, 甘, 酸, 苦, 咸, 淡, 涩] 中选择，返回数组
- meridian：从 [心, 肝, 脾, 肺, 肾, 胃, 胆, 小肠, 大肠, 膀胱, 三焦, 心包] 中选择，返回数组
- volatile：含挥发油的芳香性药材（薄荷、砂仁、川芎、当归、肉桂等）为 true
- mineral：矿物类药材（石膏、朱砂、龙骨等）为 true
- precious：贵重药材（人参、鹿茸、麝香、牛黄等）为 true
- toxic：有毒或小毒的药材（附子、半夏、细辛等）为 true
- texture：starchy=富含淀粉，fibrous=纤维性强，mucilaginous=黏腻，oily=油性，hard=坚硬（矿物），gelatinous=胶质，normal=一般

输入名称：{herb_name}

只返回 JSON，不要任何解释：
{{
    "is_herb": true或false,
    "nature": "性",
    "taste": ["味1", "味2"],
    "meridian": ["归经1"],
    "volatile": true/false,
    "mineral": true/false,
    "precious": true/false,
    "toxic": true/false,
    "texture": "质地"
}}"""

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
    valid = {k: v for k, v in cache.items() if not v.get('_invalid')}
    invalid = {k: v for k, v in cache.items() if v.get('_invalid')}
    return {
        'cached_count': len(cache),
        'valid_count': len(valid),
        'invalid_count': len(invalid),
        'cached_herbs': list(cache.keys()),
        'invalid_entries': list(invalid.keys()),
    }
