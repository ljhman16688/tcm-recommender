# -*- coding: utf-8 -*-
"""Export 3 datasets: label tables + feature tables"""
import sys, io, re, json, os, pandas as pd, numpy as np
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.makedirs('outputs/datasets', exist_ok=True)

with open('data/herb_property_dict.json','r',encoding='utf-8') as f: herb_dict = json.load(f)

# 加载 KG 并构建索引（用于生成41维KG特征）
with open('data/tcm_knowledge_graph.json', 'r', encoding='utf-8') as f:
    kg = json.load(f)
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
print(f'  KG loaded: {len(herb_name_to_forms)} herbs, {len(formula_to_form)} formulas indexed')

# 剂型列表（与 FeatureExtractor 一致）
ALL_FORMS = ['丸剂', '片剂', '胶囊剂', '颗粒剂', '口服液', '散剂', '膏剂', '糖浆剂',
             '注射剂', '酒剂', '酊剂', '栓剂', '茶剂', '丹剂', '搽剂', '洗剂',
             '气雾剂', '露剂', '膜剂']

def normalize_herb_name(name):
    name=str(name).strip(); name=re.sub(r'[（(][^)）]*[)）]','',name)
    name=re.sub(r'^(?:生|鲜|炒|酒|醋|盐|姜|蜜|焦|煅|制|熟)','',name)
    alias={'萸肉':'山茱萸','法半夏':'半夏','清半夏':'半夏','姜半夏':'半夏',
           '胆南星':'天南星','炙甘草':'甘草','炙黄芪':'黄芪',
           '熟地黄':'地黄','生地黄':'地黄','醋柴胡':'柴胡'}
    return alias.get(name,name)

def parse_rx(text):
    if pd.isna(text): return [],[]
    herbs,doses=[],[]
    pattern=r'([一-龥]{1,6}(?:[（(][^）)]+[）)])?)\s*(\d+\.?\d*)\s*g'
    for m in re.findall(pattern,str(text)):
        name=m[0].strip()
        if len(name)>=2: herbs.append(name); doses.append(float(m[1]))
    return herbs,doses

def extract_features(rows):
    flist=[]
    for _,row in rows.iterrows():
        herbs=row.get('herb_list',[]); doses=row.get('dose_list',[])
        n=len(herbs)
        if n==0: continue
        feats={'drug_name':row.get('drug_name',''),'herb_list':'、'.join(herbs),
               'indication_text':str(row.get('indication',''))[:200],
               'form':row.get('form',''),'num_herbs':n}
        nc={'寒':0,'凉':0,'平':0,'温':0,'热':0}
        tc={'辛':0,'甘':0,'酸':0,'苦':0,'咸':0,'淡':0,'涩':0}
        mc={'心':0,'肝':0,'脾':0,'肺':0,'肾':0,'胃':0,'胆':0,'小肠':0,'大肠':0,'膀胱':0,'三焦':0,'心包':0}
        xc={'hard':0,'starchy':0,'fibrous':0,'mucilaginous':0,'oily':0,'gelatinous':0,'normal':0}
        cv=ct=cm=cp=kn=0
        for herb in herbs:
            hc=normalize_herb_name(herb)
            props=herb_dict.get(herb,herb_dict.get(hc))
            if props is None: xc['normal']+=1; continue
            kn+=1
            if props.get('volatile'): cv+=1
            if props.get('toxic'): ct+=1
            if props.get('mineral'): cm+=1
            if props.get('precious'): cp+=1
            nat=props.get('nature','')
            if nat in nc: nc[nat]+=1
            for t in props.get('taste',[]):
                if t in tc: tc[t]+=1
            for m in props.get('meridian',[]):
                if m in mc: mc[m]+=1
            tex=props.get('texture','normal')
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
        ind=str(row.get('indication',''))
        if any(w in ind for w in ['急性','急症','暴发']): feats['urgency']='acute'
        elif any(w in ind for w in ['慢性','久','迁延','日久']): feats['urgency']='chronic'
        else: feats['urgency']='unspecified'
        feats['is_topical']=int(any(w in ind for w in ['外用','涂','敷','贴']))
        for k,ps in {
            'syndrome_blood_stasis':['血瘀','瘀血','化瘀','活血'],
            'syndrome_qi_deficiency':['气虚','益气','补气'],
            'syndrome_blood_deficiency':['血虚','血亏','养血','补血'],
            'syndrome_yin_deficiency':['阴虚','滋阴','养阴'],
            'syndrome_yang_deficiency':['阳虚','温阳','壮阳','补阳'],
            'syndrome_damp_heat':['湿热','清热利湿'],
            'syndrome_wind':['风寒','风热','风湿','祛风','疏风','散风'],
            'syndrome_phlegm':['化痰','祛痰','痰','豁痰'],
            'syndrome_fire_toxin':['热毒','火毒','清热解毒','泻火'],
            'syndrome_qi_stagnation':['气滞','理气','行气','疏肝'],
        }.items(): feats[k]=int(any(p in ind for p in ps))
        for k,ps in {
            'disease_respiratory':['感冒','咳嗽','喘','哮','肺','咽','喉','鼻'],
            'disease_digestive':['胃','肠','胆','泻','秘','食积','腹胀'],
            'disease_cardiovascular':['心','脑','胸痹','心悸','眩晕','中风'],
            'disease_gynecology':['月经','带下','崩漏','胎','产','子宫'],
            'disease_rheumatology':['风湿','痹','筋骨','关节','跌打','骨折'],
            'disease_dermatology':['疮','痈','疖','疹','癣','痒','斑'],
            'disease_pediatrics':['小儿','惊风','疳'],
            'disease_urology':['淋','浊','水肿'],
        }.items(): feats[k]=int(any(p in ind for p in ps))
        if len(doses)>0:
            a=np.array(doses)
            feats['total_weight']=a.sum(); feats['avg_dose']=a.mean()
            feats['max_dose']=a.max(); feats['min_dose']=a.min()
            feats['dose_std']=a.std(); feats['dose_range']=a.max()-a.min()
            feats['max_min_ratio']=a.max()/a.min() if a.min()>0 else 0
            feats['log_total_weight']=np.log1p(a.sum())
        else:
            for k in ['total_weight','avg_dose','max_dose','min_dose',
                      'dose_std','dose_range','max_min_ratio','log_total_weight']:
                feats[k]=0
        # —— KG 特征（41维）：与 FeatureExtractor.extract_features 保持一致 ——
        # 19维 药材→剂型偏好
        form_scores = {f: 0.0 for f in ALL_FORMS}
        hc = 0
        for herb in herbs:
            hc_norm = normalize_herb_name(herb)
            for kg_key in [herb, 'herb:' + herb, hc_norm, 'herb:' + hc_norm]:
                if kg_key in herb_name_to_forms:
                    for form, prob in herb_name_to_forms[kg_key].items():
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
        # 19维 适应症→剂型偏好
        ind_kw = re.findall(r'[一-龥]{2,6}', ind)
        ind_fs = Counter()
        for kw in ind_kw[:10]:
            for f in ALL_FORMS:
                ind_fs[f] += ind_to_forms.get((kw, f), 0) or ind_to_forms.get(('indication:' + kw, f), 0)
        total = sum(ind_fs.values())
        for f in ALL_FORMS:
            feats['kg_ind_form_' + f] = ind_fs[f] / max(total, 1)
        # 3维 聚合统计
        hs = [feats['kg_herb_form_' + f] for f in ALL_FORMS]
        feats['kg_herb_form_max'] = max(hs)
        feats['kg_herb_form_entropy'] = -sum(p * np.log(p + 1e-9) for p in hs if p > 0)
        feats['kg_ind_form_max'] = max([feats['kg_ind_form_' + f] for f in ALL_FORMS])
        flist.append(feats)
    return pd.DataFrame(flist)

# ======== DS1: 药典 ========
print('[1/3] Dataset 1: pharmacopoeia')
df_yp = pd.read_excel('data/药典数据.xlsx')
df_yp[['herb_list','dose_list']] = df_yp['prescription_raw'].apply(lambda x: pd.Series(parse_rx(x)))
df_yp = df_yp[df_yp['herb_list'].apply(len)>0].copy()
df_yp['indication']=df_yp['indication'].fillna(''); df_yp['form']=df_yp['dosage_form']
label1 = df_yp[['drug_name','form','indication','prescription_raw']].copy()
label1.columns = ['drug_name','form','indication','prescription_raw']
label1.to_csv('outputs/datasets/Dataset1_pharmacopoeia_labels.csv', index=False, encoding='utf-8-sig')
rows = df_yp[['herb_list','dose_list','indication','form']].copy()
rows['drug_name'] = df_yp['drug_name']
X1 = extract_features(rows)
X1.to_csv('outputs/datasets/Dataset1_pharmacopoeia_features.csv', index=False, encoding='utf-8-sig')
nfeat = len([c for c in X1.columns if c not in ['drug_name','herb_list','indication_text','form']])
print('  labels=%d  features=%d (%d dim)' % (len(label1), len(X1), nfeat))

# ======== DS2: 部颁标准 ========
print('[2/3] Dataset 2: ministerial standard')
df_s1 = pd.read_excel('data/others.xlsx', sheet_name='Sheet1')
df_test = pd.read_excel('data/others.xlsx', sheet_name='Sheet4')
test_names = set(str(n).strip() for n in df_test.iloc[:,2].dropna())
df_s1 = df_s1[~df_s1['复方名称'].isin(test_names)]

def infer_form(name, process):
    name = str(name) if pd.notna(name) else ''
    process = str(process) if pd.notna(process) else ''
    rules = [('滴丸','丸剂'),('软胶囊','胶囊剂'),('注射液','注射剂'),('口服液','口服液'),
             ('吸入剂','气雾剂'),('喷雾剂','气雾剂'),('滴眼剂','洗剂'),('滴剂','洗剂'),
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
    return 'unknown'

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
    '清艾条':'膏剂','清艾绒':'膏剂'}

df_s1['form'] = df_s1.apply(lambda r: infer_form(r['复方名称'],r['制法']), axis=1)
for name, form in manual_map.items(): df_s1.loc[df_s1['复方名称']==name, 'form'] = form
df_s1[['herb_list','dose_list']] = df_s1['含量/克数'].apply(lambda x: pd.Series(parse_rx(x)))
df_s1 = df_s1[(df_s1['form']!='unknown') & (df_s1['herb_list'].apply(len)>0)].copy()
df_s1['drug_name']=df_s1['复方名称']; df_s1['indication']=df_s1['功能与主治'].fillna('')
label2 = df_s1[['drug_name','form','indication','含量/克数','制法']].copy()
label2.columns = ['drug_name','form','indication','prescription_raw','method']
label2.to_csv('outputs/datasets/Dataset2_ministerial_labels.csv', index=False, encoding='utf-8-sig')
rows2 = df_s1[['drug_name','herb_list','dose_list','indication','form']]
X2 = extract_features(rows2)
X2.to_csv('outputs/datasets/Dataset2_ministerial_features.csv', index=False, encoding='utf-8-sig')
nfeat2 = len([c for c in X2.columns if c not in ['drug_name','herb_list','indication_text','form']])
print('  labels=%d  features=%d (%d dim)' % (len(label2), len(X2), nfeat2))

# ======== DS3: 全部合并 ========
print('[3/3] Dataset 3: merged')
X3 = pd.concat([X1, X2], ignore_index=True)
X3.to_csv('outputs/datasets/Dataset3_merged_features.csv', index=False, encoding='utf-8-sig')
label3 = pd.concat([label1[['drug_name','form','indication']], label2[['drug_name','form','indication']]], ignore_index=True)
label3.to_csv('outputs/datasets/Dataset3_merged_labels.csv', index=False, encoding='utf-8-sig')
nfeat3 = len([c for c in X3.columns if c not in ['drug_name','herb_list','indication_text','form']])
print('  labels=%d  features=%d (%d dim)' % (len(label3), len(X3), nfeat3))

print('\nDONE - files in outputs/datasets/')
