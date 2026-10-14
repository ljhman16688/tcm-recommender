# -*- coding: utf-8 -*-
"""PMI 版 KG 特征构建（2026-10-13 自 compare_kg_leakage_712.py 逐字提取，单一来源）。

compare_kg_leakage_712.py 中的同名函数为 2026-10-13 归档冻结副本（该脚本不再改动，
其八变体结果已存档 REFERENCE_*），本模块为下游（最终模型训练/部署/绘图）的规范来源。
两处唯一区别：本模块不 import 脚本级全局（ALL_FORMS/排除自环参数由调用方传入）。
"""
import math
from collections import Counter


def build_aggregates_pmi(recs, all_forms):
    """同时返回 herb->form / ind->form 计数与各自边际统计（仅从给定记录集统计）。"""
    herb_form = Counter()
    ind_form = Counter()
    for rec in recs:
        f = rec['form']
        for h in rec['herbs']:
            herb_form[(h, f)] += 1
        for kw in rec['ind_kws']:
            ind_form[(kw, f)] += 1
    marg_h, marg_f = Counter(), Counter()
    for (h, f), c in herb_form.items():
        marg_h[h] += c
        marg_f[f] += c
    Z = sum(herb_form.values())
    marg_kw, marg_f2 = Counter(), Counter()
    for (kw, f), c in ind_form.items():
        marg_kw[kw] += c
        marg_f2[f] += c
    Zind = sum(ind_form.values())
    return herb_form, ind_form, marg_h, marg_f, Z, marg_kw, marg_f2, Zind


def _pmi(c, marg_key, marg_form, Z):
    """PMI(log2) = log2( P(key,form) / (P(key)*P(form)) )；count<=0 视为无信号=0"""
    if c <= 0:
        return 0.0
    if marg_key <= 0 or marg_form <= 0 or Z <= 0:
        return 0.0
    return math.log2((c * Z) / (marg_key * marg_form))


def kg_features_pmi(herbs, ind_kws, form, agg, all_forms, exclude_self):
    """41 维 PMI 版 KG 特征。exclude_self=True 时按本样本中该药材/关键词的全部票数
    扣减（2026-10-13 LOO 修正：重复药材时原 `-1` 会残留自环票）。"""
    herb_form, ind_form, marg_h, marg_f, Z, marg_kw, marg_f2, Zind = agg
    self_cnt = Counter(herbs) if exclude_self else None
    self_kw = Counter(ind_kws) if exclude_self else None
    feats = {}
    form_scores = {f: 0.0 for f in all_forms}
    hc = 0
    for h in herbs:
        vals = []
        for f in all_forms:
            c = herb_form.get((h, f), 0)
            if exclude_self and f == form:
                c = max(0, c - self_cnt[h])
            vals.append(_pmi(c, marg_h.get(h, 0), marg_f.get(f, 0), Z))
        if any(v != 0 for v in vals):
            for i, f in enumerate(all_forms):
                form_scores[f] += vals[i]
            hc += 1
    for f in all_forms:
        feats['kg_herb_form_' + f] = form_scores[f] / hc if hc > 0 else 0.0
    ind_scores = {f: 0.0 for f in all_forms}
    nkw = 0
    for kw in ind_kws:
        vals = []
        for f in all_forms:
            c = ind_form.get((kw, f), 0)
            if exclude_self and f == form:
                c = max(0, c - self_kw[kw])
            vals.append(_pmi(c, marg_kw.get(kw, 0), marg_f2.get(f, 0), Zind))
        if any(v != 0 for v in vals):
            for i, f in enumerate(all_forms):
                ind_scores[f] += vals[i]
            nkw += 1
    for f in all_forms:
        feats['kg_ind_form_' + f] = ind_scores[f] / nkw if nkw > 0 else 0.0
    hs = [feats['kg_herb_form_' + f] for f in all_forms]
    feats['kg_herb_form_max'] = max(hs)
    feats['kg_herb_form_entropy'] = max(hs) - min(hs)  # PMI可为负，用极差替代熵
    feats['kg_ind_form_max'] = max([feats['kg_ind_form_' + f] for f in all_forms])
    return feats