# -*- coding: utf-8 -*-
"""
BERT 语义编码实验：纯文本 vs 手工特征 vs 拼接
三组对比：BERT-only / CatBoost-only / BERT+CatBoost
"""

import sys, io, re, json, pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ==============================================
# 1. 构建文本输入
# ==============================================
def build_text_input(herb_dose_list, indication_text):
    """构建 BERT 输入文本（含克数）"""
    parts = []
    for item in herb_dose_list:
        if isinstance(item, tuple):
            h, d = item
            parts.append(f'{h} {d}g')
        else:
            if len(item) >= 2:
                parts.append(item)
    herbs = '、'.join(parts)
    ind = indication_text[:200] if indication_text else ''
    return f"药材：{herbs}。适应症：{ind}"

def load_data():
    """加载全部合并数据并构建文本"""
    # 从原始数据重建
    df_yp = pd.read_excel('data/药典数据.xlsx')
    df_s1 = pd.read_excel('data/others.xlsx', sheet_name='Sheet1')

    # 药典
    def parse_rx(text):
        if pd.isna(text): return []
        herbs = []
        for m in re.findall(r'([一-龥]{1,6}(?:[（(][^）)]+[）)])?)\s*(\d+\.?\d*)\s*g', str(text)):
            if len(m[0].strip()) >= 2: herbs.append((m[0].strip(), m[1]))
        return herbs

    df_yp[['herb_list']] = df_yp['prescription_raw'].apply(lambda x: pd.Series([parse_rx(x)]))
    df_yp = df_yp[df_yp['herb_list'].apply(len) > 0].copy()
    df_yp['indication_str'] = df_yp['indication'].fillna('')
    df_yp['form'] = df_yp['dosage_form']
    df_yp['text'] = df_yp.apply(lambda r: build_text_input(r['herb_list'], r['indication_str']), axis=1)

    # 部颁标准

    def infer_form(name, process):
        name = str(name) if pd.notna(name) else ''
        process = str(process) if pd.notna(process) else ''
        rules = [('滴丸','丸剂'),('软胶囊','胶囊剂'),('注射液','注射剂'),('口服液','口服液'),
                 ('糖浆','糖浆剂'),('颗粒','颗粒剂'),('冲剂','颗粒剂'),('胶囊','胶囊剂'),
                 ('片','片剂'),('栓','栓剂'),('膏','膏剂'),('丹','丹剂'),
                 ('散','散剂'),('丸','丸剂'),('酒','酒剂'),('酊','酊剂'),
                 ('茶','茶剂'),('露','露剂'),('膜','膜剂'),('贴膏','膏剂'),
                 ('搽剂','搽剂'),('洗剂','洗剂'),('合剂','口服液'),('锭','丹剂'),
                 ('精','口服液'),('浆','糖浆剂'),('液','口服液'),('汁','口服液'),
                 ('饮','口服液'),('曲','茶剂'),('胶','膏剂'),('粉','散剂'),
                 ('水','酊剂'),('糕','茶剂'),('油','搽剂'),('宁','口服液'),
                 ('末','散剂'),('晶','颗粒剂'),('净','搽剂'),('贴','膏剂')]
        for kw, form in rules:
            if kw in name: return form
        # 手工补充
        manual = {
            '八宝眼药':'洗剂','沈阳红药':'胶囊剂','云南白药':'散剂','京万红':'膏剂',
            '桂林西瓜霜':'散剂','儿康宁':'口服液','十滴水':'酊剂','正骨水':'酊剂',
            '西瓜霜':'散剂','脉络通':'片剂','蓝花药':'丸剂','绿雪':'散剂',
            '宝宝乐':'颗粒剂','婴儿素':'散剂','湛江蛇药':'散剂','橙皮':'茶剂',
            '六神曲':'茶剂','珍珠层粉':'散剂','药墨':'丹剂','康复新液':'口服液',
            '维血宁':'口服液','引阳索':'颗粒剂','安神宁':'口服液','生乳汁':'口服液',
            '清艾条':'膏剂','清艾绒':'膏剂','按摩乳':'搽剂','胃药':'片剂',
            '顽癣净':'搽剂','口腔炎喷雾剂':'气雾剂','烧伤净喷雾剂':'气雾剂',
            '心舒静吸入剂':'气雾剂','鼻炎滴剂':'洗剂','鼻通宁滴剂':'洗剂',
            '夏天无眼药水':'洗剂','风火眼药':'洗剂','蚕茧眼药':'洗剂',
            '痔疮外洗药':'洗剂','白敬宇眼药':'膏剂','口疳吹药':'散剂',
            '吊筋药':'散剂','健脾八珍糕':'茶剂','伤科敷药':'散剂','牙痛药水':'酊剂',
            '腹痛水':'酊剂','藿香水':'酊剂','八角茴香水':'酊剂','止痒消炎水':'酊剂',
            '六神祛暑水':'酊剂','复方香薷水':'酊剂','伤友擦剂':'搽剂',
            '参贝陈皮':'茶剂','参耳五味晶':'颗粒剂','小儿疳积糖':'颗粒剂',
            '珍珠八宝眼药':'洗剂','赛空青眼药':'洗剂','保宁半夏曲':'茶剂',
            '闽东建曲':'茶剂','老范志万应神曲':'茶剂','升血调元汤':'口服液',
            '川贝半夏液':'口服液','岩果止咳液':'口服液','云南蛇药':'口服液',
            '复方蛇胆陈皮末':'散剂','蛇胆姜粒':'茶剂','药制橄榄盐':'茶剂',
            '复方铁苋止血粉':'散剂','伤痛舒':'膏剂','伤可贴':'膏剂',
            '伤科敷药':'散剂','筋骨止痛凝胶':'凝胶剂',
        }
        if name in manual: return manual[name]
        return None

    df_s1['form'] = df_s1.apply(lambda r: infer_form(r['复方名称'], r['制法']), axis=1)
    df_s1[['herb_list']] = df_s1['含量/克数'].apply(lambda x: pd.Series([parse_rx(x)]))
    df_s1 = df_s1[(df_s1['form'].notna()) & (df_s1['herb_list'].apply(len) > 0)].copy()
    df_s1['indication_str'] = df_s1['功能与主治'].fillna('')
    df_s1['text'] = df_s1.apply(lambda r: build_text_input(r['herb_list'], r['indication_str']), axis=1)

    # 合并
    df_all = pd.concat([
        df_yp[['text', 'form', 'herb_list', 'indication_str']],
        df_s1[['text', 'form', 'herb_list', 'indication_str']]
    ], ignore_index=True)

    # 剂型映射：统一非标准名称
    FORM_MAP = {'软胶囊': '胶囊剂', '滴丸': '丸剂', '贴膏剂': '膏剂'}
    df_all['form'] = df_all['form'].replace(FORM_MAP)

    # 18 种标准剂型（排除膜剂，样本数 < 5）
    STANDARD_18 = ['丸剂','片剂','胶囊剂','颗粒剂','口服液','散剂','膏剂','糖浆剂',
                   '注射剂','酒剂','酊剂','栓剂','茶剂','丹剂','搽剂','洗剂','气雾剂','露剂']
    df_all = df_all[df_all['form'].isin(STANDARD_18)].copy()

    # 标签编码
    le = LabelEncoder()
    df_all['label'] = le.fit_transform(df_all['form'])

    print(f'Total samples: {len(df_all)}, Classes: {len(le.classes_)}')
    return df_all, le

# ==============================================
# 2. BERT 模型
# ==============================================
class BertClassifier(nn.Module):
    def __init__(self, model_path, num_classes, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==============================================
# 3. 加载手工特征（用于拼接实验）
# ==============================================
def load_handcrafted_features(df_all):
    """为合并数据构建68维手工特征（匹配现有数据集格式）"""
    # 直接加载已有的特征表
    X1 = pd.read_csv('outputs/datasets/Dataset1_pharmacopoeia_features.csv')
    X2 = pd.read_csv('outputs/datasets/Dataset2_ministerial_features.csv')
    X_hf_all = pd.concat([X1, X2], ignore_index=True)

    # 剂型映射 + 18 种标准剂型
    FORM_MAP = {'软胶囊': '胶囊剂', '滴丸': '丸剂', '贴膏剂': '膏剂'}
    X_hf_all['form'] = X_hf_all['form'].replace(FORM_MAP)
    STANDARD_18 = ['丸剂','片剂','胶囊剂','颗粒剂','口服液','散剂','膏剂','糖浆剂',
                   '注射剂','酒剂','酊剂','栓剂','茶剂','丹剂','搽剂','洗剂','气雾剂','露剂']
    X_hf_all = X_hf_all[X_hf_all['form'].isin(STANDARD_18)].copy()

    le = LabelEncoder()
    labels = le.fit_transform(X_hf_all['form'])

    # 特征列
    meta_cols = ['drug_name', 'herb_list', 'indication_text', 'form']
    feat_cols = [c for c in X_hf_all.columns if c not in meta_cols]
    X = X_hf_all[feat_cols].fillna(0).copy()
    if 'urgency' in X.columns: X['urgency'] = X['urgency'].astype(str)

    return X, labels, le, feat_cols

# ==============================================
# 4. 评估函数
# ==============================================
def compute_metrics(y_true, y_pred, y_proba):
    yti = y_true
    ypi = y_pred
    yp = y_proba

    top1 = (ypi == yti).mean()
    top3 = np.any(np.argsort(yp, 1)[:, -3:] == yti[:, None], 1).mean()
    top5 = np.any(np.argsort(yp, 1)[:, -5:] == yti[:, None], 1).mean()
    ranks = np.argsort(np.argsort(-yp, 1), 1)
    mrr = (1.0 / (ranks[np.arange(len(yti)), yti] + 1)).mean()
    # NDCG@5
    top5_idx = np.argsort(yp, 1)[:, -5:][:, ::-1]
    dcg5 = np.zeros(len(yti))
    for i in range(5):
        dcg5 += (top5_idx[:, i] == yti).astype(float) / np.log2(i + 2)
    ndcg5 = dcg5.mean()  # IDCG=1 for single-label

    y_pb = np.zeros_like(yp)
    y_pb[np.arange(len(ypi)), ypi] = 1
    y_tb = np.zeros_like(yp)
    y_tb[np.arange(len(yti)), yti] = 1
    micro = f1_score(y_tb, y_pb, average='micro')
    macro = f1_score(y_tb, y_pb, average='macro')

    return top1, top3, top5, mrr, ndcg5, micro, macro

# ==============================================
# 5. BERT 训练
# ==============================================
def train_bert_model(model, train_loader, val_loader, epochs=5, lr=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_acc = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                val_acc += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_acc /= val_total
        if val_acc > best_acc:
            best_acc = val_acc
        msg = f'  Epoch {epoch+1}/{epochs} | loss={train_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}'
        print(msg, flush=True)
        with open('training_progress.log', 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()

    return model

# ==============================================
# 6. 主流程
# ==============================================
if __name__ == '__main__':
    print('=' * 60)
    print('BERT Semantic Encoding Experiment')
    print('=' * 60)

    # ===== 加载数据 =====
    df_all, le = load_data()

    # 分割
    y_idx = df_all['label'].values
    cc = pd.Series(y_idx).value_counts()
    ys = y_idx.copy()
    for rc in cc[cc < 3].index: ys[y_idx == rc] = 999

    train_texts, test_texts, y_train, y_test = train_test_split(
        df_all['text'].tolist(), y_idx, test_size=0.2, stratify=ys, random_state=42)
    train_texts, val_texts, y_train, y_val = train_test_split(
        train_texts, y_train, test_size=0.125, stratify=y_train, random_state=42)

    print(f'Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(y_test)}')

    # ===== BERT =====
    print('\n' + '=' * 60)
    print('[Experiment 1] BERT-only')
    print('=' * 60)

    # 优先本地缓存（离线模式，跳过网络检查）
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
    _ = BertModel.from_pretrained(model_name, local_files_only=True)
    model_dir = model_name  # 后续用这个名字
    train_ds = TextDataset(train_texts, y_train, tokenizer)
    val_ds = TextDataset(val_texts, y_val, tokenizer)
    test_ds = TextDataset(test_texts, y_test, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    bert_model = BertClassifier(model_dir, num_classes=len(le.classes_)).to(device)
    bert_model = train_bert_model(bert_model, train_loader, val_loader, epochs=5)

    # BERT 评估
    bert_model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = bert_model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    y_proba_bert = np.vstack(all_probs)
    y_pred_bert = np.array(all_preds)
    y_test_arr = np.array(all_labels)

    t1b, t3b, t5b, mrrb, ndcg5b, micb, macb = compute_metrics(y_test_arr, y_pred_bert, y_proba_bert)
    print(f'\nBERT-only Results:')
    print(f'  Top-1={t1b:.4f} Top-3={t3b:.4f} Top-5={t5b:.4f} NDCG@5={ndcg5b:.4f} MRR={mrrb:.4f} Micro-F1={micb:.4f} Macro-F1={macb:.4f}')

    # ===== CatBoost =====
    print('\n' + '=' * 60)
    print('[Experiment 2] CatBoost-only (handcrafted features)')
    print('=' * 60)
    X_hf, y_hf, le_hf, feat_cols = load_handcrafted_features(df_all)

    # 对齐分割
    y_idx_hf = y_hf
    cc_hf = pd.Series(y_idx_hf).value_counts()
    ys_hf = y_idx_hf.copy()
    for rc in cc_hf[cc_hf < 3].index: ys_hf[y_idx_hf == rc] = 999

    Xtr, Xt, ytr, yt = train_test_split(X_hf, y_hf, test_size=0.2, stratify=ys_hf, random_state=42)
    Xtr, Xv, ytr, yv = train_test_split(Xtr, ytr, test_size=0.125, stratify=ytr, random_state=42)

    ci = [list(feat_cols).index('urgency')] if 'urgency' in feat_cols else []
    model_cb = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        random_seed=42, loss_function='MultiClass', eval_metric='MultiClass',
        cat_features=ci, early_stopping_rounds=50, verbose=0)
    model_cb.fit(Pool(Xtr, ytr, cat_features=ci), eval_set=Pool(Xv, yv, cat_features=ci), plot=False)

    yp_cb = model_cb.predict_proba(Xt)
    ypi_cb = yp_cb.argmax(axis=1)
    t1c, t3c, t5c, mrrc, ndcg5c, micc, macc = compute_metrics(yt, ypi_cb, yp_cb)
    print(f'CatBoost-only Results:')
    print(f'  Top-1={t1c:.4f} Top-3={t3c:.4f} Top-5={t5c:.4f} NDCG@5={ndcg5c:.4f} MRR={mrrc:.4f} Micro-F1={micc:.4f} Macro-F1={macc:.4f}')

    # ===== BERT + CatBoost 拼接 =====
    print('\n' + '=' * 60)
    print('[Experiment 3] BERT + CatBoost (concatenated)')
    print('=' * 60)

    # 提取 BERT [CLS] 特征
    bert_model.eval()
    cls_features = []
    with torch.no_grad():
        for text in train_texts + val_texts + test_texts:
            enc = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
            outputs = bert_model.bert(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device))
            cls_features.append(outputs.pooler_output.cpu().numpy()[0])

    cls_arr = np.array(cls_features)
    n_train = len(train_texts)
    n_val = len(val_texts)

    # 拼接
    X_hf_numeric = X_hf[feat_cols].fillna(0).copy()
    if 'urgency' in X_hf_numeric.columns: X_hf_numeric['urgency'] = X_hf_numeric['urgency'].astype('category').cat.codes

    n_samples = min(len(cls_arr), len(X_hf_numeric))
    cls_arr = cls_arr[:n_samples]
    X_hf_numeric = X_hf_numeric.iloc[:n_samples]
    y_combined = y_hf[:n_samples]

    X_combined = np.hstack([cls_arr, X_hf_numeric.values])

    # 分割
    ys_c = y_combined.copy()
    cc_c = pd.Series(ys_c).value_counts()
    for rc in cc_c[cc_c < 3].index: ys_c[ys_c == rc] = 999

    Xtr_c, Xt_c, ytr_c, yt_c = train_test_split(X_combined, y_combined, test_size=0.2, stratify=ys_c, random_state=42)
    Xtr_c, Xv_c, ytr_c, yv_c = train_test_split(Xtr_c, ytr_c, test_size=0.125, stratify=ytr_c, random_state=42)

    model_comb = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        random_seed=42, loss_function='MultiClass', eval_metric='MultiClass',
        early_stopping_rounds=50, verbose=0)
    model_comb.fit(Xtr_c, ytr_c, eval_set=(Xv_c, yv_c), plot=False)

    yp_comb = model_comb.predict_proba(Xt_c)
    ypi_comb = yp_comb.argmax(axis=1)
    t1m, t3m, t5m, mrrm, ndcg5m, micm, macm = compute_metrics(yt_c, ypi_comb, yp_comb)
    print(f'BERT+CatBoost Results:')
    print(f'  Top-1={t1m:.4f} Top-3={t3m:.4f} Top-5={t5m:.4f} NDCG@5={ndcg5m:.4f} MRR={mrrm:.4f} Micro-F1={micm:.4f} Macro-F1={macm:.4f}')

    # ===== 最终对比 =====
    print('\n' + '=' * 80)
    print('FINAL COMPARISON')
    print('=' * 80)
    hdr = '%-25s %8s %8s %8s %8s %8s %10s %10s'
    print(hdr % ('Model', 'Top-1', 'Top-3', 'Top-5', 'NDCG@5', 'MRR', 'Micro-F1', 'Macro-F1'))
    print('-' * 85)
    print(hdr % ('BERT-only', f'{t1b:.4f}', f'{t3b:.4f}', f'{t5b:.4f}', f'{ndcg5b:.4f}', f'{mrrb:.4f}', f'{micb:.4f}', f'{macb:.4f}'))
    print(hdr % ('CatBoost-only', f'{t1c:.4f}', f'{t3c:.4f}', f'{t5c:.4f}', f'{ndcg5c:.4f}', f'{mrrc:.4f}', f'{micc:.4f}', f'{macc:.4f}'))
    print(hdr % ('BERT+CatBoost', f'{t1m:.4f}', f'{t3m:.4f}', f'{t5m:.4f}', f'{ndcg5m:.4f}', f'{mrrm:.4f}', f'{micm:.4f}', f'{macm:.4f}'))
    print('\nDONE')
