# -*- coding: utf-8 -*-
import sys, io, re, json, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
import warnings; warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('data/herb_property_dict.json','r',encoding='utf-8') as f: herb_dict = json.load(f)

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
    for m in re.findall(r'([一-龥]{1,6}?)\s*(\d+\.?\d*)\s*g',str(text)):
        if len(m[0].strip())>=2: herbs.append(m[0].strip()); doses.append(float(m[1]))
    return herbs,doses

def extract_features(rows):
    flist=[]
    for _,row in rows.iterrows():
        herbs,doses=row.get('herb_list',[]),row.get('dose_list',[])
        n=len(herbs)
        if n==0: continue
        feats={'form':row.get('form',''),'num_herbs':n}
        nc={'寒':0,'凉':0,'平':0,'温':0,'热':0}
        tc={'辛':0,'甘':0,'酸':0,'苦':0,'咸':0,'淡':0,'涩':0}
        mc={'心':0,'肝':0,'脾':0,'肺':0,'肾':0,'胃':0,'胆':0,'小肠':0,'大肠':0,'膀胱':0,'三焦':0,'心包':0}
        xc={'hard':0,'starchy':0,'fibrous':0,'mucilaginous':0,'oily':0,'gelatinous':0,'normal':0}
        cv,ct,cm,cp,kn=0,0,0,0,0
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
        for k,ps in {'syndrome_blood_stasis':['血瘀','瘀血','化瘀','活血'],
                     'syndrome_qi_deficiency':['气虚','益气','补气'],
                     'syndrome_blood_deficiency':['血虚','血亏','养血','补血'],
                     'syndrome_yin_deficiency':['阴虚','滋阴','养阴'],
                     'syndrome_yang_deficiency':['阳虚','温阳','壮阳','补阳'],
                     'syndrome_damp_heat':['湿热','清热利湿'],
                     'syndrome_wind':['风寒','风热','风湿','祛风','疏风','散风'],
                     'syndrome_phlegm':['化痰','祛痰','痰','豁痰'],
                     'syndrome_fire_toxin':['热毒','火毒','清热解毒','泻火'],
                     'syndrome_qi_stagnation':['气滞','理气','行气','疏肝']}.items():
            feats[k]=int(any(p in ind for p in ps))
        for k,ps in {'disease_respiratory':['感冒','咳嗽','喘','哮','肺','咽','喉','鼻'],
                     'disease_digestive':['胃','肠','胆','泻','秘','食积','腹胀'],
                     'disease_cardiovascular':['心','脑','胸痹','心悸','眩晕','中风'],
                     'disease_gynecology':['月经','带下','崩漏','胎','产','子宫'],
                     'disease_rheumatology':['风湿','痹','筋骨','关节','跌打','骨折'],
                     'disease_dermatology':['疮','痈','疖','疹','癣','痒','斑'],
                     'disease_pediatrics':['小儿','惊风','疳'],
                     'disease_urology':['淋','浊','水肿']}.items():
            feats[k]=int(any(p in ind for p in ps))
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
        flist.append(feats)
    return pd.DataFrame(flist)

# Load
df_yp = pd.read_excel('data/药典数据.xlsx')
df_yp[['herb_list','dose_list']] = df_yp['prescription_raw'].apply(lambda x: pd.Series(parse_rx(x)))
df_yp = df_yp[df_yp['herb_list'].apply(len)>0].copy()
df_yp['indication']=df_yp['indication'].fillna(''); df_yp['form']=df_yp['dosage_form']
rows=df_yp[['herb_list','dose_list','indication','form']]
X = extract_features(rows)
meta=['form']; feats_cols=[c for c in X.columns if c not in meta]
Xm = X[feats_cols].fillna(0).copy()
if 'urgency' in Xm.columns: Xm['urgency']=Xm['urgency'].astype(str)
y_labels=X['form'].tolist()
mlb=MultiLabelBinarizer(); y=mlb.fit_transform([[l] for l in y_labels])
yi=y.argmax(axis=1)

# Split
cc=pd.Series(yi).value_counts(); ys=yi.copy()
for rc in cc[cc<3].index: ys[yi==rc]=999
Xtv,Xt,ytv,yt=train_test_split(Xm,y,test_size=0.2,stratify=ys,random_state=42)
ytvi=ytv.argmax(axis=1); cc2=pd.Series(ytvi).value_counts(); ys2=ytvi.copy()
for rc in cc2[cc2<3].index: ys2[ytvi==rc]=999
Xtr,Xv,ytr,yv=train_test_split(Xtv,ytv,test_size=0.125,stratify=ys2,random_state=42)

# Class weights
form_counts = pd.Series(y_labels).value_counts()
total = len(y_labels)
n_classes = len(mlb.classes_)
class_weights = []
for cls in mlb.classes_:
    cnt = form_counts.get(cls, 1)
    w = total / (n_classes * cnt)
    class_weights.append(round(w, 2))

print('Class weights:')
for i, (cls, w) in enumerate(zip(mlb.classes_, class_weights)):
    cnt = form_counts.get(cls, 0)
    mark = ' ***' if w > 5 else ''
    print('  %-10s n=%-4d weight=%.2f%s' % (cls, cnt, w, mark))

ci=[feats_cols.index('urgency')]

# Train default
print('\n[Default] Training...')
m1=CatBoostClassifier(iterations=1000,learning_rate=0.05,depth=6,l2_leaf_reg=3,
    random_seed=42,loss_function='MultiClass',eval_metric='MultiClass',
    cat_features=ci,early_stopping_rounds=50,verbose=0)
m1.fit(Pool(Xtr,ytr.argmax(axis=1),cat_features=ci),eval_set=Pool(Xv,yv.argmax(axis=1),cat_features=ci),plot=False)
yp1=m1.predict_proba(Xt); ypi1=yp1.argmax(axis=1)
ypb1=np.zeros_like(yt); ypb1[np.arange(len(ypi1)),ypi1]=1

# Train weighted
print('[Weighted] Training...')
m2=CatBoostClassifier(iterations=1000,learning_rate=0.05,depth=6,l2_leaf_reg=3,
    random_seed=42,loss_function='MultiClass',eval_metric='MultiClass',
    class_weights=class_weights,cat_features=ci,early_stopping_rounds=50,verbose=0)
m2.fit(Pool(Xtr,ytr.argmax(axis=1),cat_features=ci),eval_set=Pool(Xv,yv.argmax(axis=1),cat_features=ci),plot=False)
yp2=m2.predict_proba(Xt); ypi2=yp2.argmax(axis=1)
ypb2=np.zeros_like(yt); ypb2[np.arange(len(ypi2)),ypi2]=1

# Metrics
yti=yt.argmax(axis=1)
def metrics(yp, ypb, yti, yt):
    pi=yp.argmax(axis=1)
    t1=(pi==yti).mean()
    t3=np.any(np.argsort(yp,1)[:,-3:]==yti[:,None],1).mean()
    t5=np.any(np.argsort(yp,1)[:,-5:]==yti[:,None],1).mean()
    ranks=np.argsort(np.argsort(-yp,1),1)
    mr=(1.0/(ranks[np.arange(len(yti)),yti]+1)).mean()
    mic=f1_score(yt,ypb,average='micro')
    mac=f1_score(yt,ypb,average='macro')
    return t1,t3,t5,mr,mic,mac

d1=metrics(yp1,ypb1,yti,yt); d2=metrics(yp2,ypb2,yti,yt)

print()
print('='*65)
print('Per-class F1 comparison')
print('='*65)
hdr = '%s %10s %10s %10s' % ('Form','Default','Weighted','Delta')
print(hdr)
print('-'*45)
for i, cls in enumerate(mlb.classes_):
    f1d = f1_score(yt[:,i], ypb1[:,i], zero_division=0)
    f1w = f1_score(yt[:,i], ypb2[:,i], zero_division=0)
    delta = f1w - f1d
    if f1d > 0 or f1w > 0:
        mark = ' <<' if abs(delta) > 0.03 else ''
        print('%s %10.4f %10.4f %+10.4f%s' % (cls, f1d, f1w, delta, mark))

print()
hdr2 = '%s %10s %10s %10s' % ('Overall','Default','Weighted','Delta')
print(hdr2)
print('-'*45)
for name,a,b in [('Top-1',d1[0],d2[0]),('Top-3',d1[1],d2[1]),('Top-5',d1[2],d2[2]),
                  ('MRR',d1[3],d2[3]),('Micro-F1',d1[4],d2[4]),('Macro-F1',d1[5],d2[5])]:
    print('%s %10.4f %10.4f %+10.4f' % (name,a,b,b-a))
print('\nDONE')
