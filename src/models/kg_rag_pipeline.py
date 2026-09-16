# -*- coding: utf-8 -*-
"""
KG-RAG 推理流水线：分类器 + KG检索 → Prompt → LLM可解释推荐
"""

import sys, os, io, re, json, pandas as pd, numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from openai import OpenAI
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')

# Import from extract_features (single source of truth for herb normalization + features)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.features.extract_features import normalise_herb

# ==============================================
# 1. Load resources
# ==============================================
print('='*60)
print('Loading resources...')
print('='*60)

with open('data/herb_property_dict.json','r',encoding='utf-8') as f: herb_dict = json.load(f)
with open('data/tcm_knowledge_graph.json','r',encoding='utf-8') as f: kg = json.load(f)

id2ent = {int(eid): info['name'] for eid, info in kg['entities'].items()}
herb_to_formulas = {}; formula_to_form = {}; formula_to_name = {}
for t in kg['triples']:
    if t['relation_name'] == 'contained_in':
        herb_to_formulas.setdefault(t['head'], []).append(t['tail'])
    elif t['relation_name'] == 'has_form':
        formula_to_form[t['head']] = t['tail_name']
        formula_to_name[t['head']] = t['head_name']

ALL_FORMS = ['丸剂','片剂','胶囊剂','颗粒剂','口服液','散剂','膏剂','糖浆剂',
             '注射剂','酒剂','酊剂','栓剂','茶剂','丹剂','搽剂','洗剂',
             '气雾剂','露剂','膜剂']

# Build KG lookup indices
herb_name_to_forms = {}
herb_form_count = Counter()
for h_id, f_ids in herb_to_formulas.items():
    h_name = id2ent.get(h_id, str(h_id))
    fc = Counter()
    for f_id in f_ids:
        f = formula_to_form.get(f_id)
        if f:
            fc[f] += 1
            h_name_clean = re.sub(r'herb:', '', h_name)
            herb_form_count[(h_name_clean, f)] += 1
    if fc:
        t = sum(fc.values())
        herb_name_to_forms[h_name] = {k: v/t for k, v in fc.most_common()}

ind_to_forms = Counter()
form_to_herbs = {}  # dosage_form -> common herbs
for t in kg['triples']:
    if t['relation_name'] == 'treats':
        f = formula_to_form.get(t['head'])
        if f:
            ind_to_forms[(t['tail_name'], f)] += 1
    # Build form->herbs index
    if t['relation_name'] == 'contained_in' and t['tail'] in formula_to_form:
        form = formula_to_form[t['tail']]
        if form not in form_to_herbs: form_to_herbs[form] = Counter()
        form_to_herbs[form][id2ent.get(t['head'],'')] += 1

print('  KG: %d herbs in lookup, %d form mappings' % (len(herb_name_to_forms), len(formula_to_form)))

# ==============================================
# 2. Train CatBoost classifier (quick)
# ==============================================
print('\nTraining CatBoost classifier...')

def parse_rx(text):
    if pd.isna(text): return [], []
    herbs, doses = [], []
    for m in re.findall(r'([一-龥]{1,6}(?:[（(][^）)]+[）)])?)\s*(\d+\.?\d*)\s*(?:g|ml|mg)', str(text)):
        if len(m[0].strip()) >= 2: herbs.append(m[0].strip()); doses.append(float(m[1]))
    return herbs, doses


# LLM Fallback
LLM_CACHE_FILE = 'data/herb_llm_cache.json'
def _load_llm_cache():
    if os.path.exists(LLM_CACHE_FILE):
        with open(LLM_CACHE_FILE,'r',encoding='utf-8') as f: return json.load(f)
    return {}
def _save_llm_cache(cache):
    with open(LLM_CACHE_FILE,'w',encoding='utf-8') as f: json.dump(cache, f, ensure_ascii=False, indent=2)
def _query_llm_for_herb(herb_name):
    try:
        from openai import OpenAI
        client = OpenAI(api_key='sk-zk2faa63c8d59e691b199fcdb2873b1624569afdfb598afe', base_url='https://api.zhizengzeng.com/v1')
        resp = client.chat.completions.create(
            model='qwen3.7-max',
            messages=[{'role':'system','content':'你是中药学专家。只输出JSON: {"nature":"","taste":[],"meridian":[],"volatile":false,"toxic":false,"mineral":false,"precious":false,"texture":"normal"}'},
                      {'role':'user','content':herb_name}],
            temperature=0, max_tokens=150)
        r = json.loads(resp.choices[0].message.content)
        return {'nature':r.get('nature',''),'taste':[t for t in r.get('taste',[]) if t in {'辛','甘','酸','苦','咸','淡','涩','微苦','微甘'}],
                'meridian':[m for m in r.get('meridian',[]) if m in {'心','肝','脾','肺','肾','胃','胆','小肠','大肠','膀胱','三焦','心包'}],
                'volatile':bool(r.get('volatile',False)),'toxic':bool(r.get('toxic',False)),
                'mineral':bool(r.get('mineral',False)),'precious':bool(r.get('precious',False)),
                'texture':r.get('texture','normal') if r.get('texture','normal') in {'hard','starchy','fibrous','mucilaginous','oily','gelatinous','normal'} else 'normal'}
    except: return None
def get_herb_props(herb_name):
    hc = normalise_herb(herb_name)
    if herb_name in herb_dict: return herb_dict[herb_name]
    if hc in herb_dict: return herb_dict[hc]
    cache = _load_llm_cache()
    if hc in cache: return cache[hc]
    if herb_name in cache: return cache[herb_name]
    print('  [LLM] querying: %s' % herb_name)
    props = _query_llm_for_herb(herb_name)
    if props: cache[herb_name] = props; _save_llm_cache(cache)
    return props

def build_features(herbs, doses, indication):
    n = len(herbs)
    if n == 0: return None
    feats = {'num_herbs': n}
    nc = {'寒':0,'凉':0,'平':0,'温':0,'热':0}
    tc = {'辛':0,'甘':0,'酸':0,'苦':0,'咸':0,'淡':0,'涩':0}
    mc = {'心':0,'肝':0,'脾':0,'肺':0,'肾':0,'胃':0,'胆':0,'小肠':0,'大肠':0,'膀胱':0,'三焦':0,'心包':0}
    xc = {'hard':0,'starchy':0,'fibrous':0,'mucilaginous':0,'oily':0,'gelatinous':0,'normal':0}
    cv=ct=cm=cp=kn=0
    for herb in herbs:
        hc = normalise_herb(herb)
        props = get_herb_props(herb)
        if props is None: xc['normal']+=1; continue
        kn += 1
        if props.get('volatile'): cv+=1
        if props.get('toxic'): ct+=1
        if props.get('mineral'): cm+=1
        if props.get('precious'): cp+=1
        nat = props.get('nature','')
        if nat in nc: nc[nat]+=1
        for t in props.get('taste',[]):
            if t in tc: tc[t]+=1
        for m in props.get('meridian',[]):
            if m in mc: mc[m]+=1
        tex = props.get('texture','normal')
        if tex in xc: xc[tex]+=1
    feats['has_volatile']=int(cv>0); feats['has_toxic']=int(ct>0)
    feats['has_mineral']=int(cm>0); feats['has_precious']=int(cp>0)
    feats['volatile_ratio']=cv/n; feats['toxic_ratio']=ct/n
    feats['mineral_ratio']=cm/n; feats['precious_ratio']=cp/n; feats['known_ratio']=kn/n
    for nat in ['寒','凉','平','温','热']: feats['nature_'+nat+'_ratio']=nc[nat]/n
    for t in ['辛','甘','酸','苦','咸','淡','涩']: feats['taste_'+t+'_ratio']=tc[t]/n
    for m in ['心','肝','脾','肺','肾','胃','胆','小肠','大肠','膀胱','三焦','心包']:
        feats['meridian_'+m+'_ratio']=mc[m]/n
    for tex in ['hard','starchy','fibrous','mucilaginous','oily','gelatinous']:
        feats['texture_'+tex+'_ratio']=xc[tex]/n
    ind = str(indication) if indication else ''
    if any(w in ind for w in ['急症','暴发']): feats['urgency']='acute'
    elif any(w in ind for w in ['慢性','久','迁延']): feats['urgency']='chronic'
    else: feats['urgency']='unspecified'
    feats['is_topical']=int(any(w in ind for w in ['外用','涂','敷','贴']))
    for k,ps in {
        'syndrome_blood_stasis':['血瘀','化瘀','活血'],
        'syndrome_qi_deficiency':['气虚','益气','补气'],
        'syndrome_yin_deficiency':['阴虚','滋阴','养阴'],
        'syndrome_wind':['风寒','风热','风湿','祛风','疏风'],
        'syndrome_phlegm':['化痰','祛痰','豁痰'],
        'syndrome_fire_toxin':['热毒','火毒','清热解毒','泻火'],
        'syndrome_qi_stagnation':['气滞','理气','行气','疏肝'],
    }.items(): feats[k]=int(any(p in ind for p in ps))
    for k,ps in {
        'disease_respiratory':['感冒','咳嗽','喘','哮','咽','喉'],
        'disease_digestive':['胃','肠','胆','泻','秘','食积'],
        'disease_cardiovascular':['胸痹','心悸','眩晕','中风'],
        'disease_gynecology':['月经','带下','崩漏','胎'],
        'disease_rheumatology':['风湿','痹','筋骨','关节'],
        'disease_dermatology':['疮','痈','疹','癣'],
        'disease_pediatrics':['小儿','惊风'],
        'disease_urology':['淋','浊','水肿'],
    }.items(): feats[k]=int(any(p in ind for p in ps))
    if doses:
        a=np.array(doses)
        feats['total_weight']=a.sum(); feats['avg_dose']=a.mean()
        feats['max_dose']=a.max(); feats['min_dose']=a.min()
        feats['dose_std']=a.std(); feats['dose_range']=a.max()-a.min()
        feats['max_min_ratio']=a.max()/a.min() if a.min()>0 else 0
        feats['log_total_weight']=np.log1p(a.sum())
    else:
        for k in ['total_weight','avg_dose','max_dose','min_dose',
                  'dose_std','dose_range','max_min_ratio','log_total_weight']: feats[k]=0
    # KG features
    form_scores = {f: 0.0 for f in ALL_FORMS}
    hc = 0
    for herb in herbs:
        for kg_key in [herb, 'herb:'+herb]:
            if kg_key in herb_name_to_forms:
                for form, prob in herb_name_to_forms[kg_key].items():
                    if form in form_scores: form_scores[form] += prob
                hc += 1; break
    if hc > 0:
        for f in ALL_FORMS: feats['kg_herb_form_'+f] = form_scores[f] / hc
    else:
        for f in ALL_FORMS: feats['kg_herb_form_'+f] = 0.0
    ind_kw = re.findall(r'[一-龥]{2,6}', str(indication) if indication else '')
    ind_fs = Counter()
    for kw in ind_kw[:10]:
        for f in ALL_FORMS: ind_fs[f] += ind_to_forms.get((kw, f), 0)
    total = sum(ind_fs.values())
    for f in ALL_FORMS: feats['kg_ind_form_'+f] = ind_fs[f] / max(total, 1)
    hs = [feats['kg_herb_form_'+f] for f in ALL_FORMS]
    feats['kg_herb_form_max'] = max(hs)
    feats['kg_herb_form_entropy'] = -sum(p*np.log(p+1e-9) for p in hs if p>0)
    feats['kg_ind_form_max'] = max([feats['kg_ind_form_'+f] for f in ALL_FORMS])

    return feats

# Build training data
rows = []
df_yp = pd.read_excel('data/药典数据.xlsx')
for _, r in df_yp.iterrows():
    herbs, doses = parse_rx(r['prescription_raw'])
    if not herbs: continue
    ind = r['indication'] if pd.notna(r['indication']) else ''
    feats = build_features(herbs, doses, ind)
    if feats: feats['form'] = r['dosage_form']; rows.append(feats)

df_s1 = pd.read_excel('data/others.xlsx', sheet_name='Sheet1')
xl = pd.ExcelFile('data/others.xlsx')
df_test_names = set(str(n).strip() for n in pd.read_excel('data/others.xlsx', sheet_name=xl.sheet_names[3]).iloc[:,2].dropna())
df_s1 = df_s1[~df_s1.iloc[:,3].isin(df_test_names)]

def infer_form(name, process):
    name = str(name) if pd.notna(name) else ''; process = str(process) if pd.notna(process) else ''
    rules = [('滴丸','丸剂'),('软胶囊','胶囊剂'),('注射液','注射剂'),('口服液','口服液'),
             ('糖浆','糖浆剂'),('颗粒','颗粒剂'),('胶囊','胶囊剂'),
             ('片','片剂'),('栓','栓剂'),('膏','膏剂'),('丹','丹剂'),
             ('散','散剂'),('丸','丸剂'),('酒','酒剂'),('酊','酊剂'),
             ('茶','茶剂'),('露','露剂'),('膜','膜剂'),('贴膏','膏剂'),
             ('合剂','口服液'),('锭','丹剂'),('精','口服液'),('浆','糖浆剂'),
             ('液','口服液'),('曲','茶剂'),('胶','膏剂'),('粉','散剂')]
    for kw, form in rules:
        if kw in name: return form
    return None

manual = {'八宝眼药':'洗剂','桂林西瓜霜':'散剂','京万红':'膏剂','沈阳红药':'胶囊剂',
    '云南白药':'散剂','儿康宁':'口服液','十滴水':'酊剂','正骨水':'酊剂',
    '西瓜霜':'散剂','脉络通':'片剂','蓝花药':'丸剂','绿雪':'散剂'}

for _, r in df_s1.iterrows():
    drug = str(r.iloc[3]) if pd.notna(r.iloc[3]) else ''
    raw = str(r.iloc[4]) if pd.notna(r.iloc[4]) else ''
    method = str(r.iloc[6]) if pd.notna(r.iloc[6]) else ''
    ind = str(r.iloc[7]) if pd.notna(r.iloc[7]) else ''
    form = infer_form(drug, method)
    if form is None and drug in manual: form = manual[drug]
    if form is None: continue
    herbs, doses = parse_rx(raw)
    if not herbs: continue
    feats = build_features(herbs, doses, ind)
    if feats: feats['form'] = form; rows.append(feats)

df_all = pd.DataFrame(rows)
fc = df_all['form'].value_counts()
df_all = df_all[df_all['form'].isin(fc[fc>=5].index)].copy()
le = LabelEncoder(); y = le.fit_transform(df_all['form'])

hc_cols = [c for c in df_all.columns if not c.startswith('kg_') and c != 'form']
kg_cols = [c for c in df_all.columns if c.startswith('kg_')]
X_train = df_all[hc_cols + kg_cols].fillna(0).copy()
urg_map = {'acute':0, 'chronic':1, 'unspecified':2}
if 'urgency' in X_train.columns: X_train['urgency'] = X_train['urgency'].map(urg_map).fillna(2)

model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    random_seed=42, loss_function='MultiClass', verbose=0)
model.fit(X_train, y)
print('  Classifier trained: %d samples, %d classes, %d features' % (len(X_train), len(le.classes_), X_train.shape[1]))

# Save model and metadata to web_app/
import pickle
model.save_model('web_app/catboost_model.cbm')
with open('web_app/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
feat_cols = list(X_train.columns)
with open('web_app/feature_cols.pkl', 'wb') as f:
    pickle.dump(feat_cols, f)
print('  Model saved to web_app/ (%d features)' % len(feat_cols))

# ==============================================
# 3. KG Retriever + Rule Engine
# ==============================================
print('\nBuilding KG Retriever + Rule Engine...')

# Load dosage form rules
with open('data/dosage_form_rules.json', 'r', encoding='utf-8') as f:
    rules_db = json.load(f)['rules']
print('  Loaded %d dosage form selection rules' % len(rules_db))

def evaluate_trigger(trigger, herb_list, features):
    """Evaluate whether a rule's trigger condition is met."""
    ttype = trigger['type']

    if ttype == 'herb_match':
        # Check if any herb in the list matches the trigger herbs
        target_herbs = set(trigger['herbs'])
        mode = trigger.get('match_mode', 'any')
        matched = [h for h in herb_list if normalise_herb(h) in target_herbs or h in target_herbs]
        if mode == 'any':
            return len(matched) > 0, matched
        elif mode == 'all':
            return len(matched) == len(target_herbs), matched
        return False, []

    elif ttype == 'property_check':
        field = trigger['field']
        val = trigger['value']
        if field == 'has_volatile':
            return any(herb_dict.get(h, herb_dict.get(normalise_herb(h), {})).get('volatile') for h in herb_list), []
        elif field == 'has_toxic':
            return any(herb_dict.get(h, herb_dict.get(normalise_herb(h), {})).get('toxic') for h in herb_list), []
        elif field == 'has_mineral':
            return any(herb_dict.get(h, herb_dict.get(normalise_herb(h), {})).get('mineral') for h in herb_list), []
        elif field == 'has_precious':
            return any(herb_dict.get(h, herb_dict.get(normalise_herb(h), {})).get('precious') for h in herb_list), []
        return False, []

    elif ttype == 'texture_check':
        field = trigger['field']
        op = trigger.get('op', 'gt')
        threshold = trigger['value']
        n = features.get('num_herbs', 1)
        if n == 0: return False, []
        # Count herbs with matching texture
        texture_map = {'starchy': 0, 'fibrous': 0, 'mucilaginous': 0, 'oily': 0, 'gelatinous': 0}
        for h in herb_list:
            props = herb_dict.get(h, herb_dict.get(normalise_herb(h), {}))
            t = props.get('texture', 'normal')
            if t in texture_map: texture_map[t] += 1
        ratio = texture_map.get(field.replace('_ratio', ''), 0) / n
        if op == 'gt': return ratio > threshold, []
        elif op == 'gte': return ratio >= threshold, []
        return False, []

    elif ttype == 'feature_check':
        field = trigger['field']
        op = trigger.get('op', 'eq')
        value = trigger['value']
        fval = features.get(field, None)
        if fval is None: return False, []
        if op == 'eq': return fval == value, []
        elif op == 'gte': return fval >= value, []
        elif op == 'gt': return fval > value, []
        return False, []

    elif ttype == 'form_check':
        # Used in composite triggers only - check if target form is being considered
        return True, []

    elif ttype == 'composite':
        op = trigger['op']
        results = [evaluate_trigger(c, herb_list, features)[0] for c in trigger['conditions']]
        if op == 'and': return all(results), []
        elif op == 'or': return any(results), []
        return False, []

    return False, []

def match_rules(herb_list, features):
    """Match all rules against the given herb list and features.
    Returns rules grouped by evidence_level."""
    matched = {'L1': [], 'L2': [], 'L3': [], 'L4': []}

    for rule in rules_db:
        triggered, matched_herbs = evaluate_trigger(rule['trigger'], herb_list, features)
        if triggered:
            # Format recommendation with matched herbs
            rec = rule['recommendation']
            if matched_herbs:
                rec = rec.replace('{herb}', '、'.join(matched_herbs[:3]))
            level_key = 'L%d' % rule['evidence_level']
            matched[level_key].append({
                'id': rule['id'],
                'category': rule['category'],
                'evidence_level': rule['evidence_level'],
                'rule_name': rule['rule_name'],
                'action': rule['action'],
                'target_forms': rule.get('target_forms', []),
                'boost_weight': rule.get('boost_weight', 0),
                'recommendation': rec,
                'source': rule['source'],
                'exceptions': rule['exceptions'],
                'expert_consensus': rule['expert_consensus'],
                'certainty': rule['certainty']
            })

    return matched

def retrieve_evidence(herb_list, indication):
    """从KG检索推理证据 + 规则匹配"""
    evidence = {'herb_properties': [], 'similar_formulas': [], 'disease_prefs': [], 'rules': []}

    # 1. Herb properties
    for herb in herb_list:
        props = {}
        for kg_key in [herb, 'herb:'+herb]:
            if kg_key in herb_name_to_forms:
                top3 = sorted(herb_name_to_forms[kg_key].items(), key=lambda x:-x[1])[:3]
                top3_str = ', '.join(['%s(%.0f%%)' % (f, p*100) for f, p in top3])
                props[herb] = top3_str
                break
        # Check herb dict for nature/taste/meridian
        hc = normalise_herb(herb)
        hinfo = herb_dict.get(herb, herb_dict.get(hc, {}))
        if hinfo.get('nature') or hinfo.get('taste'):
            tastes = ','.join(hinfo.get('taste',[]))
            nature = hinfo.get('nature','')
            meridians = ','.join(hinfo.get('meridian',[]))
            props['nature'] = '%s, %s, 归%s经' % (tastes, nature, meridians)

        if props:
            evidence['herb_properties'].append(props)

    # 2. Similar formulas (find formulas containing these herbs)
    formula_scores = Counter()
    for herb in herb_list:
        # Find herb name in KG: look through herb_name_to_forms + also search by entity ID
        for kg_key in [herb, 'herb:'+herb]:
            # Search herb_to_formulas by herb name via id2ent reverse lookup
            for eid, ename in id2ent.items():
                if ename == kg_key and eid in herb_to_formulas:
                    for f_id in herb_to_formulas[eid]:
                        if f_id in formula_to_form:
                            formula_scores[f_id] += 1
    top_formulas = formula_scores.most_common(5)
    for f_id, score in top_formulas:
        f_name = formula_to_name.get(f_id, str(f_id))
        f_name = re.sub(r'formula:', '', f_name)
        f_form = formula_to_form.get(f_id, '')
        evidence['similar_formulas'].append({'name': f_name, 'form': f_form, 'overlap': score})

    # 3. Disease-form preferences
    ind_kw = re.findall(r'[一-龥]{2,6}', str(indication) if indication else '')
    disease_prefs = Counter()
    for kw in ind_kw[:10]:
        for f in ALL_FORMS:
            cnt = ind_to_forms.get((kw, f), 0)
            if cnt > 5: disease_prefs[f] += cnt
    if disease_prefs:
        total = sum(disease_prefs.values())
        evidence['disease_prefs'] = [(f, cnt/total) for f, cnt in disease_prefs.most_common(5)]

    # 4. Dosage form selection rules (from rule engine)
    # Build features dict for rule matching
    _feats = build_features(herb_list, [], indication) or {}
    matched_rules = match_rules(herb_list, _feats)

    # Format rules for prompt, grouped by evidence level
    rule_lines = []
    level_labels = {1: '法规禁忌（硬约束）', 2: '风险警示', 3: '工艺调整建议', 4: '弱先验'}
    for level in [1, 2, 3, 4]:
        level_rules = matched_rules.get('L%d' % level, [])
        if level_rules:
            rule_lines.append('【%s】' % level_labels[level])
            for i, r in enumerate(level_rules, 1):
                source_info = ' [来源: %s | 专家一致性: %s]' % (r['source'], r['expert_consensus'])
                rule_lines.append('  %d. [%s] %s%s' % (i, r['rule_name'], r['recommendation'], source_info))
                if r['exceptions']:
                    rule_lines.append('     例外: %s' % '; '.join(r['exceptions']))

    evidence['rules'] = rule_lines
    evidence['matched_rules'] = matched_rules  # Store structured rules for other uses
    return evidence

# ==============================================
# 4. Prompt Builder
# ==============================================
SYSTEM_PROMPT = """你是一位资深中药制剂专家。根据药材组成、适应症和提供的知识依据，推荐最合适的5种中药剂型并给出详细推理过程。

可选剂型：丸剂、片剂、胶囊剂、颗粒剂、口服液、散剂、膏剂、糖浆剂、注射剂、酒剂、酊剂、栓剂、茶剂、丹剂、搽剂、洗剂、气雾剂、露剂、膜剂

剂型选择规则按证据强度分为四级：
- 法规禁忌（L1）：具有法律法规或药典依据的硬约束，必须遵守
- 风险警示（L2）：提示潜在风险，需审慎评估，不直接排除剂型
- 工艺调整建议（L3）：基于制剂工艺经验，可通过工艺优化规避
- 弱先验（L4）：基于历史用药统计，仅作为排序参考，不作为硬性条件

请按以下JSON格式输出，不要输出其他内容：
{
  "recommendations": [
    {"rank": 1, "form": "剂型名", "confidence": 0.XX, "reason": "推理依据"},
    {"rank": 2, "form": "剂型名", "confidence": 0.XX, "reason": "推理依据"},
    ...
  ],
  "excluded": [
    {"form": "剂型名", "reason": "排除原因"}
  ]
}"""

def build_prompt(herb_list, indication, top5_forms, evidence):
    """构建KG增强的Prompt"""
    herbs_str = '、'.join(herb_list)

    # Classifier results
    classifier_str = '\n'.join(['  %d. %s (%.3f)' % (i+1, f, c) for i, (f, c) in enumerate(top5_forms)])

    # Herb properties
    prop_str = ''
    for p in evidence['herb_properties']:
        herb_name = [k for k in p.keys() if k != 'nature'][0] if any(k != 'nature' for k in p.keys()) else ''
        nature_info = p.get('nature', '')
        form_pref = p.get(herb_name, '') if herb_name else ''
        line = '  %s: %s' % (herb_name, nature_info)
        if form_pref:
            line += ' | KG历史剂型偏好: %s' % form_pref
        prop_str += line + '\n'

    # Similar formulas
    sim_str = ''
    for f in evidence['similar_formulas'][:5]:
        sim_str += '  - %s（%s，药材重叠%d味）\n' % (f['name'], f['form'], f['overlap'])

    # Disease preferences
    dis_str = ''
    for f, p in evidence.get('disease_prefs', [])[:5]:
        dis_str += '  - %s (%.0f%%)\n' % (f, p*100)

    # Rules
    rule_str = '\n'.join(evidence['rules']) if evidence['rules'] else '无'

    prompt = """【药材组成】%s

【适应症】%s

【分类器推荐 Top-5】
%s

【知识图谱检索依据】

药材属性与历史剂型偏好：
%s
相似处方及剂型：
%s
适应症-剂型偏好：
%s

剂型选择规则（按证据强度分级，L1为硬约束，L4为弱先验）：
%s

请根据以上信息，推荐5种最适合的剂型并说明推理过程。""" % (herbs_str, indication, classifier_str, prop_str, sim_str, dis_str, rule_str)

    return prompt

# ==============================================
# 5. Main RAG Pipeline
# ==============================================
def predict_and_explain(herb_list, indication, use_llm=True):
    """完整RAG流水线：分类 + 检索 + 推理"""
    print()
    print('='*70)
    print('INPUT')
    print('='*70)
    print('  Herbs: %s' % '、'.join(herb_list))
    print('  Indication: %s' % indication)

    # Step 1: Classifier
    herbs_raw = [h.strip() for h in herb_list if h.strip()]
    feats = build_features(herbs_raw, [], indication)
    if feats is None:
        print('ERROR: No valid herbs')
        return

    X_input = pd.DataFrame([feats]).reindex(columns=X_train.columns, fill_value=0)
    if 'urgency' in X_input.columns: X_input['urgency'] = X_input['urgency'].map(urg_map).fillna(2)

    probs = model.predict_proba(X_input)[0]
    top_idx = np.argsort(probs)[::-1][:5]
    top5 = [(le.classes_[i], probs[i]) for i in top_idx]

    # === KG Anchor: if exact herb combo exists in KG, move its form to rank 1 ===
    herb_set = frozenset(herbs_raw)
    kg_match_form = None
    kg_match_name = None
    for f_id, f_name in formula_to_name.items():
        # Get all herbs in this formula
        f_herbs = set()
        for h_id, f_ids in herb_to_formulas.items():
            if f_id in f_ids:
                h_name = id2ent.get(h_id, '')
                h_clean = re.sub(r'herb:', '', h_name)
                f_herbs.add(h_clean)
        # Check if input herbs match formula herbs (>=80% overlap, >=3 herbs)
        if len(f_herbs) >= 3 and len(herb_set & f_herbs) >= min(len(herb_set), len(f_herbs)) * 0.8:
            f_form = formula_to_form.get(f_id, '')
            clean_name = re.sub(r'formula:', '', f_name)
            f_form_clean = re.sub(r'dosage_form:', '', f_form)
            if f_form_clean in ALL_FORMS or f_form_clean in [re.sub(r'dosage_form:', '', formula_to_form.get(fid, '')) for fid in formula_to_form]:
                kg_match_form = f_form_clean
                kg_match_name = clean_name
                break

    if kg_match_form:
        # Move the KG-matched form to rank 1
        top5_forms = [f for f, _ in top5]
        if kg_match_form not in top5_forms:
            # Replace last place
            top5[-1] = (kg_match_form, probs[list(le.classes_).index(kg_match_form)] if kg_match_form in le.classes_ else 0.95)
        # Sort: KG match first, then by confidence
        top5.sort(key=lambda x: (0 if x[0] == kg_match_form else 1, -x[1]))
        print()
        print('  *** KG ANCHOR: exact match found: %s -> %s -> moved to rank 1' % (kg_match_name, kg_match_form))

    print()
    print('='*70)
    print('PHASE 1: CLASSIFIER OUTPUT')
    print('='*70)
    for rank, (form, conf) in enumerate(top5, 1):
        bar = '#' * int(conf * 50)
        print('  %d. %-8s %.3f %s' % (rank, form, conf, bar))

    # Step 2: KG Retrieval
    evidence = retrieve_evidence(herbs_raw, indication)

    print()
    print('='*70)
    print('PHASE 2: KG RETRIEVAL')
    print('='*70)
    print('  Herb properties found: %d' % len(evidence['herb_properties']))
    print('  Similar formulas: %d' % len(evidence['similar_formulas']))
    for f in evidence['similar_formulas'][:5]:
        if f['overlap'] >= 2:
            print('    %s (%s, %d herbs overlap)' % (f['name'], f['form'], f['overlap']))
    print('  Disease preferences: %d' % len(evidence.get('disease_prefs', [])))
    print('  Rules triggered: %d' % len(evidence['rules']))
    for r in evidence['rules']:
        print('    - %s' % r)
    for f in evidence['similar_formulas'][:3]:
        print('    Similar: %s (%s, overlap=%d herbs)' % (f['name'], f['form'], f['overlap']))

    # Step 3: LLM Generation
    prompt = build_prompt(herbs_raw, indication, top5, evidence)

    if use_llm:
        print()
        print('='*70)
        print('PHASE 3: LLM REASONING')
        print('='*70)

        try:
            client = OpenAI(
                api_key='sk-zk2faa63c8d59e691b199fcdb2873b1624569afdfb598afe',
                base_url='https://api.zhizengzeng.com/v1'
            )
            resp = client.chat.completions.create(
                model='qwen3.7-max',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
                max_tokens=1024,
            )
            print(resp.choices[0].message.content)
        except Exception as e:
            print('  LLM API error: %s' % str(e)[:100])
            print()
            print('  === Prompt (would be sent to LLM) ===')
            print(prompt[:500] + '...')
    else:
        print()
        print('='*70)
        print('PROMPT (LLM disabled)')
        print('='*70)
        print(prompt[:800] + '...')

    return top5, evidence

# ==============================================
# 6. Run examples
# ==============================================
if __name__ == '__main__':
    examples = [
        (['丹参','三七','冰片'], '冠心病心绞痛，气滞血瘀，胸痹'),
        (['大黄','牵牛子','槟榔','人参','朱砂'], '脾胃不和，痰食阻滞，积滞便秘'),
        (['麻黄','桂枝','苦杏仁','甘草'], '外感风寒表实证，恶寒发热，喘咳'),
        (['人参','白术','茯苓','甘草'], '脾胃气虚，面色萎黄，气短乏力，食少便溏'),
    ]

    for herbs, ind in examples:
        predict_and_explain(herbs, ind, use_llm=True)
        print()
