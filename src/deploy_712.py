# -*- coding: utf-8 -*-
"""712 口径部署推理共享模块（2026-10-14）。

背景：web_app 旧部署加载的 catboost_model.cbm 是旧管线内联训练的模型（18 类、
旧 KG 计数均值特征、真熵），与论文主模型 G 不一致。本模块统一部署端推理口径：

  - 特征构建与 experiments/leakage/train_final_model_G.py 的 build_pmi_matrix 单行逐字一致：
    68 维手工（protocol_712.handcrafted_features，merged 1826 词典、无在线 LLM 兜底）
    + 41 维 PMI-KG（kg_features_pmi，exclude_self=False，仅用冻结的训练集聚合统计）；
    urgency 数值化 {acute:0, chronic:1, unspecified:2}，fillna(0)。
  - 冻结产物（由 scripts/utils/freeze_deploy_assets.py 生成并端到端验证 Top-1=0.4515）：
      data/deploy/pmic_agg_712.json  仅训练集(3573)聚合统计，JSON 序列化
      web_app/catboost_G.cbm         主模型 G（109 维 / 19 类）

部署端对"词典外新处方"不做在线 LLM 兜底特征推断——与训练口径一致；
LLM 兜底仅出现在推理层的规则引擎/证据展示（不进入分类器特征），见 web_app/app.py。
"""
import json
from collections import Counter

from src.data.protocol_712 import handcrafted_features, ind_keywords
from src.features.extract_features import ALL_FORMS
from src.features.kg_pmi_features import kg_features_pmi

URGENCY_MAP = {'acute': 0, 'chronic': 1, 'unspecified': 2}


def save_agg(agg, path, n_train=3573):
    """build_aggregates_pmi 返回值 → JSON（tuple 键序列化为 [h, f, c] 列表）。"""
    herb_form, ind_form, marg_h, marg_f, Z, marg_kw, marg_f2, Zind = agg
    payload = {
        'version': '712', 'n_train': n_train,
        'herb_form': [[h, f, c] for (h, f), c in herb_form.items()],
        'ind_form': [[k, f, c] for (k, f), c in ind_form.items()],
        'marg_h': dict(marg_h), 'marg_f': dict(marg_f), 'Z': Z,
        'marg_kw': dict(marg_kw), 'marg_f2': dict(marg_f2), 'Zind': Zind,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    print('已冻结聚合统计 -> %s（训练 %d 条，herb_form %d / ind_form %d）' % (
        path, n_train, len(herb_form), len(ind_form)))


def load_agg(path):
    """JSON → build_aggregates_pmi 返回结构（供 kg_features_pmi 直接用）。"""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return (Counter({(h, f): c for h, f, c in d['herb_form']}),
            Counter({(k, f): c for k, f, c in d['ind_form']}),
            Counter(d['marg_h']), Counter(d['marg_f']), int(d['Z']),
            Counter(d['marg_kw']), Counter(d['marg_f2']), int(d['Zind']))


def build_row(herbs, doses, indication, admin, agg):
    """单条处方 → 109 维特征 dict（与 train_final_model_G.build_pmi_matrix 单行一致）。

    推理端 exclude_self=False：新处方不在训练聚合中，天然无自环；
    form 传 ''（仅 exclude_self=True 时被使用）。
    """
    feats = handcrafted_features(herbs, doses, indication, admin)
    feats.update(kg_features_pmi(herbs, ind_keywords(indication), '', agg, ALL_FORMS, False))
    feats['urgency'] = URGENCY_MAP.get(feats['urgency'], 2)
    return feats