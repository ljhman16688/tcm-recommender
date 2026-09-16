# -*- coding: utf-8 -*-
"""将部颁标准(Sheet1)处方数据补充到知识图谱中"""
import json, re, pandas as pd
from collections import Counter

# ===== 1. 加载现有 KG =====
with open('data/tcm_knowledge_graph.json', 'r', encoding='utf-8') as f:
    kg = json.load(f)

# 获取当前最大 ID
ent2id = kg['entities']   # dict: id -> {'name': 'herb:大黄', 'type': 'herb'}
rel2id = kg['relations']  # dict: id -> {'name': 'has_taste'}
triples = kg['triples']   # list of dicts with head/relation/tail (int IDs)
max_ent_id = max(int(k) for k in ent2id.keys())

# 预计算 relation name -> ID 映射
rel_name_to_id = {}
for rid_str, rinfo in rel2id.items():
    rel_name_to_id[rinfo['name']] = int(rid_str)

# 也构建已有三元组索引，避免重复
existing_triple_set = set()
for t in triples:
    existing_triple_set.add((t['head'], t['relation'], t['tail']))

print('Existing KG: %d entities, %d triples' % (len(ent2id), len(triples)))

# ===== 2. 加载部颁标准数据 =====
df = pd.read_excel('data/others.xlsx', sheet_name='Sheet1')
print('Sheet1 rows: %d' % len(df))

# 剂型推断(完整版)
def infer_form(name, process):
    name = str(name) if pd.notna(name) else ''
    process = str(process) if pd.notna(process) else ''
    rules = [('滴丸','丸剂'),('软胶囊','胶囊剂'),('注射液','注射剂'),('口服液','口服液'),
             ('吸入剂','气雾剂'),('喷雾剂','喷雾剂'),('滴眼剂','洗剂'),('滴剂','洗剂'),
             ('糖浆','糖浆剂'),('颗粒','颗粒剂'),('冲剂','颗粒剂'),('胶囊','胶囊剂'),
             ('片','片剂'),('栓','栓剂'),('膏','膏剂'),('丹','丹剂'),
             ('散','散剂'),('丸','丸剂'),('酒','酒剂'),('酊','酊剂'),
             ('茶','茶剂'),('露','露剂'),('膜','膜剂'),('气雾','气雾剂'),
             ('贴膏','膏剂'),('搽剂','搽剂'),('洗剂','洗剂'),
             ('合剂','口服液'),('锭','丹剂'),('精','口服液'),('浆','糖浆剂'),
             ('蜜','膏剂'),('滋','口服液'),('灵','颗粒剂'),('油','搽剂'),
             ('液','口服液'),('汁','口服液'),('饮','口服液'),('汤','口服液'),
             ('曲','茶剂'),('胶','膏剂'),('粉','散剂'),('墨','丹剂'),
             ('条','膏剂'),('绒','膏剂'),('乳','搽剂'),('净','搽剂'),
             ('贴','膏剂'),('宁','口服液'),('糕','茶剂'),('雪','散剂'),
             ('末','散剂'),('晶','颗粒剂'),('水','酊剂')]
    for kw, form in rules:
        if kw in name: return form
    p_rules = [(r'制粒|整粒','颗粒剂'),(r'装.*胶囊','胶囊剂'),(r'压.*片','片剂'),
               (r'(?:制成|泛).*丸','丸剂'),(r'制成.*锭','丹剂'),(r'模成块','茶剂'),
               (r'分装','散剂'),(r'灌装|灌封','口服液'),(r'涂布|涂膜','膏剂'),
               (r'发酵','茶剂'),(r'制成.*1000ml','口服液'),
               (r'粉碎成细粉.*混匀.*即得','散剂'),(r'加入凡士林','膏剂'),
               (r'渗漉.*乙醇.*即得','酊剂')]
    for pat, form in p_rules:
        if re.search(pat, process): return form
    return None

manual_map = {
    '八宝眼药':'洗剂','沈阳红药':'胶囊剂','云南白药':'散剂','海洋胃药':'片剂',
    '京万红':'膏剂','伤痛舒':'膏剂','桂林西瓜霜':'散剂','儿康宁':'口服液',
    '鼻通宁滴剂':'洗剂','口腔炎喷雾剂':'气雾剂','烧伤净喷雾剂':'气雾剂',
    '鼻炎滴剂':'洗剂','心舒静吸入剂':'气雾剂','按摩乳':'搽剂',
    '伤可贴':'膏剂','筋骨止痛凝胶':'凝胶剂','复方铁苋止血粉':'散剂',
    '药墨':'丹剂','珍珠层粉':'散剂','顽癣净':'搽剂','伤友擦剂':'搽剂',
    '蛇胆姜粒':'茶剂','橙皮':'茶剂','药制橄榄盐':'茶剂','维血宁':'口服液',
    '引阳索':'颗粒剂','安神宁':'口服液','生乳汁':'口服液','升血调元汤':'口服液',
    '川贝半夏液':'口服液','岩果止咳液':'口服液','云南蛇药':'口服液',
    '康复新液':'口服液','保宁半夏曲':'茶剂','闽东建曲':'茶剂',
    '老范志万应神曲':'茶剂','六神曲':'茶剂','小儿疳积糖':'颗粒剂',
    '牙痛药水':'酊剂','十滴水':'酊剂','风火眼药':'洗剂','蚕茧眼药':'洗剂',
    '痔疮外洗药':'洗剂','白敬宇眼药':'膏剂','绿雪':'散剂','口疳吹药':'散剂',
    '吊筋药':'散剂','健脾八珍糕':'茶剂','伤科敷药':'散剂','珍珠八宝眼药':'洗剂',
    '婴儿素':'散剂','腹痛水':'酊剂','赛空青眼药':'洗剂','藿香水':'酊剂',
    '宝宝乐':'颗粒剂','复方蛇胆陈皮末':'散剂','八角茴香水':'酊剂',
    '止痒消炎水':'酊剂','参贝陈皮':'茶剂','参耳五味晶':'颗粒剂',
    '湛江蛇药':'散剂','六神祛暑水':'酊剂','正骨水':'酊剂','夏天无眼药水':'洗剂',
    '西瓜霜':'散剂','复方香薷水':'酊剂','脉络通':'片剂','蓝花药':'丸剂',
    '清艾条':'膏剂','清艾绒':'膏剂',
}

def get_or_create_ent(name, etype, ent2id, max_id):
    """获取或创建实体ID"""
    for eid, info in ent2id.items():
        if info.get('name') == name:
            return int(eid), max_id
    max_id += 1
    ent2id[str(max_id)] = {'name': name, 'type': etype}
    return max_id, max_id

def parse_herbs(text):
    if pd.isna(text): return []
    herbs = []
    pattern = r'([一-龥]{1,6}(?:[（(][^）)]+[）)])?)\s*(\d+\.?\d*)\s*g'
    for m in re.findall(pattern, str(text)):
        name = m[0].strip()
        if len(name) >= 2: herbs.append(name)
    return herbs

# ===== 3. 处理每条处方 =====
new_triples = 0
new_entities = 0
formula_count = 0

for _, row in df.iterrows():
    formula_name = str(row['复方名称']) if pd.notna(row['复方名称']) else ''
    if not formula_name or formula_name == 'nan': continue

    # 去括号中的内容作为标准名
    formula_clean = re.sub(r'[（(][^）)）]*[）)]', '', formula_name).strip()

    # 剂型
    form = infer_form(formula_name, str(row['制法']) if pd.notna(row['制法']) else '')
    if form is None:
        if formula_name in manual_map:
            form = manual_map[formula_name]
        else:
            continue

    # 创建 formula 实体
    fname = 'formula:' + formula_clean
    fid, max_ent_id = get_or_create_ent(fname, 'formula', ent2id, max_ent_id)
    if fid == max_ent_id: new_entities += 1
    formula_count += 1

    # has_form 关系
    df_name = form  # 直接用剂型名称
    dfid, max_ent_id = get_or_create_ent(df_name, 'dosage_form', ent2id, max_ent_id)
    if dfid == max_ent_id: new_entities += 1
    rid = rel_name_to_id['has_form']
    if (fid, rid, dfid) not in existing_triple_set:
        triples.append({'head': fid, 'relation': rid, 'tail': dfid,
                        'head_name': fname, 'relation_name': 'has_form', 'tail_name': form})
        existing_triple_set.add((fid, rid, dfid))
        new_triples += 1

    # 药材
    herb_text = str(row['含量/克数']) if pd.notna(row['含量/克数']) else ''
    herbs = parse_herbs(herb_text)
    if not herbs:
        herb_text = str(row['处方']) if pd.notna(row['处方']) else ''
        herbs = [h.strip() for h in str(herb_text).split('、') if len(h.strip()) >= 2]

    for herb in herbs:
        hname = 'herb:' + herb
        hid, max_ent_id = get_or_create_ent(hname, 'herb', ent2id, max_ent_id)
        if hid == max_ent_id: new_entities += 1
        rid = rel_name_to_id['contained_in']
        if (hid, rid, fid) not in existing_triple_set:
            triples.append({'head': hid, 'relation': rid, 'tail': fid,
                            'head_name': hname, 'relation_name': 'contained_in', 'tail_name': fname})
            existing_triple_set.add((hid, rid, fid))
            new_triples += 1

    # 适应症
    ind_text = str(row['功能与主治']) if pd.notna(row['功能与主治']) else ''
    if ind_text and ind_text != 'nan':
        # 简单提取关键词
        inds = re.findall(r'[一-龥]{2,6}(?:证|病|痛|肿|炎|咳|喘|痹|滞|瘀|虚|热|寒|泻|秘)', ind_text)
        for ind in inds[:5]:
            iname = 'indication:' + ind
            iid, max_ent_id = get_or_create_ent(iname, 'indication', ent2id, max_ent_id)
            if iid == max_ent_id: new_entities += 1
            rid = rel_name_to_id['treats']
            if (fid, rid, iid) not in existing_triple_set:
                triples.append({'head': fid, 'relation': rid, 'tail': iid,
                                'head_name': fname, 'relation_name': 'treats', 'tail_name': iname})
                existing_triple_set.add((fid, rid, iid))
                new_triples += 1

# ===== 4. 更新统计 =====
kg['num_entities'] = len(ent2id)
kg['num_triples'] = len(triples)

# 重新统计 entity types
etype_counts = Counter()
for eid, info in ent2id.items():
    etype = info.get('type', 'unknown')
    etype_counts[etype] += 1

kg['statistics']['entity_types'] = dict(etype_counts)
kg['statistics']['relation_types']['contained_in'] = sum(1 for t in triples if t['relation_name'] == 'contained_in')
kg['statistics']['relation_types']['has_form'] = sum(1 for t in triples if t['relation_name'] == 'has_form')
kg['statistics']['relation_types']['treats'] = sum(1 for t in triples if t['relation_name'] == 'treats')

# ===== 5. 保存 =====
with open('data/tcm_knowledge_graph.json', 'w', encoding='utf-8') as f:
    json.dump(kg, f, ensure_ascii=False, indent=2)

print()
print('Added: %d formulas, %d new entities, %d new triples' % (formula_count, new_entities, new_triples))
print('Updated KG: %d entities, %d triples' % (kg['num_entities'], kg['num_triples']))
print('Entity types:')
for k, v in sorted(kg['statistics']['entity_types'].items(), key=lambda x: -x[1]):
    print('  %s: %d' % (k, v))
print('Done')
