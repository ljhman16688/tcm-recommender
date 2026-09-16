# -*- coding: utf-8 -*-
"""
LLM对比实验v2: 与CatBoost同一测试集，全量994条
Fast version: use pre-split indices
"""
import sys, io, re, json, pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI

# ==============================================
# 1. Fast data loading: use pre-exported CSVs
# ==============================================
print('Loading data...')
X1 = pd.read_csv('outputs/datasets/Dataset1_pharmacopoeia_features.csv')
X2 = pd.read_csv('outputs/datasets/Dataset2_ministerial_features.csv')
df_feat = pd.concat([X1, X2], ignore_index=True)

fc = df_feat['form'].value_counts()
df_feat = df_feat[df_feat['form'].isin(fc[fc>=5].index)].copy()
le = LabelEncoder()
y = le.fit_transform(df_feat['form'])

# Build texts from herb_list + indication_text
texts = []
for _, r in df_feat.iterrows():
    herbs = str(r.get('herb_list',''))[:200]
    ind = str(r.get('indication_text',''))[:200]
    texts.append('药材：%s。适应症：%s' % (herbs, ind))

# Same split as CatBoost
cc = pd.Series(y).value_counts(); ys = y.copy()
for rc in cc[cc<3].index: ys[y==rc]=999
_, _, _, y_test, texts_train, texts_test = train_test_split(
    df_feat.drop(columns=['drug_name','herb_list','indication_text','form']),
    y, texts, test_size=0.2, stratify=ys, random_state=42)

print('Test set: %d samples, %d classes' % (len(y_test), len(le.classes_)))
print('Form distribution: %s' % dict(pd.Series(le.inverse_transform(y_test)).value_counts().head(5)))

# ==============================================
# 2. LLM evaluation
# ==============================================
SYSTEM_PROMPT = """你是资深中药制剂专家。根据药材和适应症，推荐5种最适合的中药剂型。

可选剂型：丸剂、片剂、胶囊剂、颗粒剂、口服液、散剂、膏剂、糖浆剂、注射剂、酒剂、酊剂、栓剂、茶剂、丹剂、搽剂、洗剂、气雾剂、露剂、膜剂

只输出JSON: {"recommendations":[{"rank":1,"form":"剂型名"},{"rank":2,"form":"剂型名"},{"rank":3,"form":"剂型名"},{"rank":4,"form":"剂型名"},{"rank":5,"form":"剂型名"}]}"""

ALL_FORMS = ['丸剂','片剂','胶囊剂','颗粒剂','口服液','散剂','膏剂','糖浆剂','注射剂','酒剂','酊剂','栓剂','茶剂','丹剂','搽剂','洗剂','气雾剂','露剂','膜剂']

def parse_response(text):
    if not text: return []
    # Remove reasoning blocks (推理模型如 MiniMax-M3 会输出 <think>...</think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove markdown code block markers
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)
    # Try to find JSON with recommendations
    try:
        # Find the outermost braces containing recommendations
        m = re.search(r'\{[^{}]*"recommendations"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return [r['form'] for r in data.get('recommendations', []) if r.get('form') in ALL_FORMS][:5]
    except: pass
    # Fallback: search for form names in text
    found = []
    for f in ALL_FORMS:
        if f in text and f not in found: found.append(f)
    return found[:5]

def evaluate_llm(model_id, display_name, test_texts, test_labels):
    client = OpenAI(api_key='sk-zk2faa63c8d59e691b199fcdb2873b1624569afdfb598afe',
                    base_url='https://api.zhizengzeng.com/v1')
    total = len(test_texts)
    top1 = top3 = top5 = 0; ndcg_sum = 0.0; mrr_sum = 0.0
    per_class_hits = {}

    for i, (text, true_label) in enumerate(zip(test_texts, test_labels)):
        true_form = le.inverse_transform([true_label])[0]
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':text}],
                temperature=0, max_tokens=6000,
            )
            content = resp.choices[0].message.content
            if content is None: content = ''
            # Handle encoding issues
            content = content.encode('utf-8', errors='replace').decode('utf-8')
            pred = parse_response(content)
        except Exception as e:
            pred = []

        if true_form in pred[:1]: top1 += 1
        if true_form in pred[:3]: top3 += 1
        if true_form in pred[:5]: top5 += 1
        try:
            rank = pred.index(true_form)+1
            mrr_sum += 1.0/rank
            ndcg_sum += 1.0/np.log2(rank+1)
        except: pass
        if true_form not in per_class_hits: per_class_hits[true_form] = []
        per_class_hits[true_form].append(1 if true_form in pred[:5] else 0)

        if (i+1) % 50 == 0:
            print('  [%s] %d/%d Top1=%.3f' % (display_name, i+1, total, top1/(i+1)))

    results = {'top1': top1/total, 'top3': top3/total, 'top5': top5/total,
               'ndcg': ndcg_sum/total, 'mrr': mrr_sum/total,
               'per_class_recall': {f: sum(h)/len(h) for f, h in per_class_hits.items()}}
    print('  [%s] DONE: Top1=%.4f Top5=%.4f NDCG=%.4f MRR=%.4f' % (display_name, results['top1'], results['top5'], results['ndcg'], results['mrr']))
    return results

# ==============================================
# 3. Run all models
# ==============================================
models = [
    ('gpt-4o', 'GPT-4o'),
    ('qwen3.7-max', 'Qwen-3.7'),
    ('gemini-3.1-pro-preview', 'Gemini-3.1'),
    ('deepseek-v4-pro', 'DeepSeek-V4'),
    ('glm-5.1', 'GLM-5.1'),
    ('minimax-m3', 'MiniMax-M3'),
]

print('\nEvaluating %d LLMs on %d test samples...' % (len(models), len(y_test)))

# 断点续跑：加载已有结果，跳过已完成的模型
import os
RESULT_PATH = 'data/llm_benchmark_v2.json'
all_results = {}
if os.path.exists(RESULT_PATH):
    try:
        all_results = json.load(open(RESULT_PATH, encoding='utf-8'))
        print('Loaded %d completed model(s) from %s' % (len(all_results), RESULT_PATH))
    except Exception as e:
        print('Failed to load existing results:', e)

for model_id, display_name in models:
    if display_name in all_results:
        print('  [%s] SKIP (already done)' % display_name)
        continue
    all_results[display_name] = evaluate_llm(model_id, display_name, texts_test, y_test)
    # 每完成一个模型立即增量落盘，防止中断丢失
    json.dump(all_results, open(RESULT_PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('  [%s] saved (cumulative %d/%d)' % (display_name, len(all_results), len(models)))

# ==============================================
# 4. Final table
# ==============================================
print('\n' + '='*65)
print('FINAL COMPARISON (same %d test samples as CatBoost)' % len(y_test))
print('='*65)
print('%-20s %8s %8s %8s %8s %8s %8s' % ('Model', 'Top-1', 'Top-3', 'Top-5', 'NDCG', 'MRR', 'R@5'))
print('-'*65)
for name, r in sorted(all_results.items(), key=lambda x: -x[1]['top1']):
    avg_recall = np.mean(list(r['per_class_recall'].values())) if r['per_class_recall'] else 0
    print('%-20s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' % (name, r['top1'], r['top3'], r['top5'], r['ndcg'], r['mrr'], avg_recall))
print('%-20s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' % ('CatBoost (ours)', 0.5553, 0.8495, 0.9272, 0.7131, 0.7131, 0.4155))

json.dump(all_results, open('data/llm_benchmark_v2.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nSaved to data/llm_benchmark_v2.json')
print('DONE')