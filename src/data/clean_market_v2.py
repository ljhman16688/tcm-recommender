# -*- coding: utf-8 -*-
import sys, io, re, json, os, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Find file by scanning directory to avoid path encoding issues
data_dir = 'data'
target_file = None
for f in os.listdir(data_dir):
    if '清洗后' in f and f.endswith('.xlsx'):
        target_file = os.path.join(data_dir, f)
        break
if not target_file:
    for f in os.listdir(data_dir):
        if '清洗' in f and '上市' in f and f.endswith('.xlsx'):
            target_file = os.path.join(data_dir, f)
            break

print('Using file:', repr(target_file))
df = pd.read_excel(target_file)

print('Columns:', list(df.columns))

with open('data/herb_property_dict.json','r',encoding='utf-8') as f:
    herb_dict = json.load(f)
herb_names = set(herb_dict.keys())

def is_chemical(name):
    pats = [r'[a-zA-Z]{2,}', r'\d', r'苷$', r'酮$', r'酚$', r'碱$', r'酸$',
            r'素$', r'提取物', r'浸膏', r'浓缩', r'流浸膏', r'提取液',
            r'化合物', r'总皂苷', r'色素', r'香精', r'防腐剂']
    return any(re.search(p, name) for p in pats)

def clean_strict(text):
    if pd.isna(text) or str(text).strip() == '': return None

    text = re.sub(r'[。，,;；]\s*辅料[为是：:].*$', '', str(text))
    text = re.sub(r'[。，,;；]\s*辅料$', '', str(text))

    skip_words = {'辅料','蔗糖','淀粉','糊精','硬脂酸镁','滑石粉','蜂蜜','冰糖',
                  '黄酒','白酒','乙醇','水','纯化水','明胶','甘油','凡士林',
                  '蜂蜡','苯甲酸钠','羟苯乙酯','聚山梨酯','甜菊素','阿斯巴甜',
                  '山梨酸钾','柠檬酸钠','薄荷脑','苯甲酸','碳酸钙','氧化锌',
                  '甘露醇','液体石蜡','羊毛脂','丙二醇','聚乙二醇','大豆磷脂',
                  '蔗糖脂肪酸酯','单糖浆','矫味剂','微晶纤维素','羧甲淀粉钠',
                  '预胶化淀粉','交联聚维酮','羟丙甲纤维素','聚丙烯酸树脂',
                  '二氧化钛','十二烷基硫酸钠','聚维酮','黄明胶','豆油','米酒',
                  '矫味剂','香精','色素','防腐剂','蔗糖','冰糖','白酒','黄酒',
                  '蔗糖','淀粉','蜂蜜','硬脂酸镁','滑石粉','糊精'}
    herb_col = '药材_清洗后' if '药材_清洗后' in df.columns else '药材'

    herbs_raw = [h.strip().rstrip('。.') for h in re.split(r'[、，,;；。]', text) if h.strip()]
    clean_herbs = []
    for h in herbs_raw:
        if len(h) < 2: continue
        if h in skip_words: continue
        if is_chemical(h): continue
        clean_herbs.append(h)

    return '、'.join(clean_herbs) if len(clean_herbs) >= 1 else None

# Determine which column has herbs
herb_col = '药材_清洗后' if '药材_清洗后' in df.columns else '药材'
print('Using herb column:', herb_col)

df['药材_纯中药_v2'] = df[herb_col].apply(clean_strict)

before = len(df)
df_clean = df[df['药材_纯中药_v2'].notna()].copy()
print('Before: %d -> After: %d (removed %d)' % (before, len(df_clean), before - len(df_clean)))

# Check leftovers
left1 = df_clean[df_clean['药材_纯中药_v2'].str.contains('辅料', na=False)]
left2 = df_clean[df_clean['药材_纯中药_v2'].str.contains('提取|浸膏|浓缩粉|提取液|流浸膏', na=False)]
left3 = df_clean[df_clean['药材_纯中药_v2'].str.contains(r'[a-zA-Z]{3}|苷$|酮$|酚$|碱$', na=False)]
print('Leftover 辅料=%d 提取物=%d 化学=%d' % (len(left1), len(left2), len(left3)))

# Show some samples
for i in [0, 2, 10]:
    r = df_clean.iloc[i]
    print('[%d] %s' % (i, r['名称']))
    print('    %s' % str(r['药材_纯中药_v2'])[:120])

out_cols = [c for c in ['名称','剂型','适应症','药材_纯中药_v2'] if c in df_clean.columns]
df_clean[out_cols].to_excel('data/上市中药_纯中药组方.xlsx', index=False)
print()
print('Saved: data/上市中药_纯中药组方.xlsx (%d rows)' % len(df_clean))
