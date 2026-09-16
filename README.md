# 中药剂型智能推荐 (KG-RAR)

基于知识图谱检索增强的中药剂型推荐模型。输入药材列表 + 克数 + 适应症，输出 Top-5 剂型推荐 + 推理依据。

**版本**: v2.2.0 | **规则数**: 74条 | **KG证据**: 四路检索

## 评测指标

| 指标 | 值 |
|------|-----|
| Top-1 | 63.40% |
| Top-3 | 85.0% |
| Top-5 | 95.9% |
| MRR | 0.713 |
| Macro-F1 | 0.530 |
| vs. 最强LLM零样本 (Gemini-3.1) | 45.1% (+18.3pp) |

## 项目结构

```
tcm-recommender/
├── README.md
├── requirements.txt
├── .gitignore
│
├── web_app/                          # Flask Web 应用（主入口）
│   ├── app.py                        # 主应用：6步管线 + API
│   ├── catboost_model.cbm            # 训练好的 CatBoost 模型（20类）
│   ├── feature_cols.pkl              # 特征列名（109维）
│   ├── label_encoder.pkl             # 标签编码器
│   ├── static/                       # CSS / 静态资源
│   └── templates/                    # HTML 模板
│
├── src/                              # 核心源码
│   ├── features/
│   │   ├── extract_features.py       # 特征提取（68维手工 + 41维KG）
│   │   └── llm_herb_property.py      # LLM 药材属性三级查询
│   ├── data/
│   │   ├── build_herb_dict.py        # 构建药材属性字典
│   │   ├── fix_textbook.py           # 解析中药学教材
│   │   ├── export_datasets.py        # 导出特征表 CSV
│   │   ├── extend_kg.py              # 扩展知识图谱
│   │   └── clean_market_v2.py        # 清洗上市中药数据
│   ├── models/
│   │   ├── unified_ablation_v2.py    # 主训练 + 消融实验
│   │   ├── benchmark_llms_v2.py      # LLM 零样本评测
│   │   ├── kg_rag_pipeline.py        # KG-RAG 推理流水线
│   │   ├── dosage_ablation.py        # 剂量特征消融
│   │   ├── class_weight_test.py      # 类别权重实验
│   │   ├── final_compare.py          # 多源数据对比
│   │   └── train_bert.py             # BERT 对比基线
│   └── viz/
│       ├── draw_roadmap.py           # 技术路线图
│       ├── draw_kg_viz.py            # KG 可视化
│       ├── plot_metrics.py           # 指标对比图
│       ├── plot_ml_comparison.py     # ML 模型对比图
│       └── plot_all_figures.py       # 批量出图
│
├── data/                             # 数据文件
│   ├── tcm_knowledge_graph.json      # 知识图谱（16K实体, 71K三元组）
│   ├── dosage_form_rules.json        # 规则引擎（74条，四级）
│   ├── herb_property_dict.json       # 药材属性字典（687味）
│   ├── herb_llm_cache.json           # LLM 药材推断缓存（2,083条）
│   ├── training_data_full.json       # 训练数据
│   ├── textbook_herbs.json           # 教材解析结果
│   ├── llm_benchmark_v2.json         # LLM 评测结果
│   ├── data_parser.py                # 数据解析工具
│   ├── real_data_loader.py           # 数据加载工具
│   ├── test_sets/                    # 测试集 Excel 文件
│   │   ├── 测试集_最终1000.xlsx
│   │   ├── 测试集_含剂量.xlsx
│   │   ├── 测试集_重名排除.xlsx
│   │   ├── market_clean_v5.xlsx
│   │   ├── market_clean_v5_独立测试集.xlsx
│   │   └── ...
│   ├── external/                     # 外部参考数据
│   │   ├── 药典数据.xlsx
│   │   ├── 上市中药.xlsx
│   │   ├── 中药学（新世纪第五版）.md
│   │   └── ...
│   └── 中药成方制剂/                  # 成方制剂解析数据 + PDF
│
├── outputs/                          # 产出文件
│   ├── 技术路线_KG-RAR.md             # 技术路线文档
│   ├── 技术路线_KG-RAR.html           # 交互式管线图
│   ├── 案例报告_十全大补汤_KG-RAR.md   # 案例分析报告
│   ├── paper/                        # 论文各版本
│   │   ├── 论文初稿-0910.docx
│   │   ├── 论文初稿-0910_v2.2.0_修订版_v2.docx
│   │   ├── 论文大纲.md
│   │   └── ...
│   ├── figures/                      # 论文图表
│   │   ├── fig0_technical_roadmap.png
│   │   ├── fig1_ablation.png
│   │   ├── fig2_llm_compare.png
│   │   ├── fig3_dosage.png
│   │   ├── fig4_ml_comparison.png
│   │   ├── fig5_ml_all_metrics.png
│   │   ├── case-studies.html
│   │   └── ...
│   ├── datasets/                     # 特征表 CSV（用于论文复现）
│   └── polishing_20260908/           # 论文润色工作文件
│
├── scripts/                          # 脚本工具
│   ├── demos/                        # 演示和案例
│   │   ├── demo_shiquandabu.py
│   │   ├── demo_two_cases.py
│   │   └── case_study_for_paper.md
│   ├── tests/                        # 测试脚本
│   │   ├── test_rule_engine.py
│   │   └── check_balance.py
│   └── utils/                        # 工具脚本
│       ├── predict.py                # 命令行推理
│       ├── clean_cache.py            # LLM 缓存清理
│       ├── heatmap.py                # 热力图
│       ├── radar_chart.py            # 雷达图
│       └── export_to_word.py         # 论文导出
│
├── experiments/                      # 实验结果
│   ├── ablation/                     # 消融实验
│   ├── model_comparison/             # ML模型对比
│   ├── dosage/                       # 剂量消融
│   └── market_eval/                  # 上市中药评测
│
├── archive/                          # 归档（旧版代码/历史探索）
│   ├── kg_recommendation/            # KGAT/对比学习探索
│   ├── kg_construction/              # KG 构建脚本
│   ├── explanation/                  # 早期推理模块
│   ├── eda_history/                  # EDA 历史产物
│   └── *.py                          # 各版本旧脚本
│
├── logs/                             # 日志文件
├── utils/                            # 基础工具（config, metrics）
└── catboost_info/                    # CatBoost 训练产物
```

## 核心模型（v2.2.0）

| 组件 | 说明 |
|------|------|
| 特征工程 | 109维 = 手工68维 + KG特征41维（19 herb-form + 19 ind-form + 3 stats） |
| 分类器 | CatBoost，20类（18种基础剂型 + 滴丸、软胶囊），交叉熵损失 |
| 规则引擎 | 74条四级规则：L1法规禁忌(8) / L2风险警示(16) / L3工艺建议(25) / L4弱先验(25) |
| KG锚定 | 双层：①101首经典名方精确匹配 → ②KG处方精确匹配 |
| KG-RAG | 四路KG检索（药材偏好 + 相似处方 + 疾病关联 + 规则匹配）→ 模板推理 + 可选LLM增强 |
| 药材查询 | 三级：herb_property_dict(687) → herb_llm_cache(2,083) → LLM实时兜底 |
| 冲突检测 | exclude_vs_boost / warn_vs_boost |

## 六步推理管线

```
Step 1  特征提取 → 109维融合特征
Step 2  CatBoost推理 → 18种剂型初始概率
Step 3  规则引擎匹配 → 74条规则逐条求值，L1/L2/L3/L4分组
Step 4  概率融合 → L1置零 → 归一化 → 排序 → 双层锚定 → 冲突检测
Step 5  KG四路证据检索 → 药材偏好/相似处方/疾病关联/规则匹配
Step 6  推理生成 → 模板推理 + 可选 LLM (qwen3.7-max)
```

## 部署

- **Web应用**: `python web_app/app.py` → http://localhost:5000
- **命令行**: `python scripts/utils/predict.py --herbs "人参 10g, 白术 15g" --indication "虚劳"`
- **离线模式**: 纯 Python + JSON 数据，0 Token 消耗
- **增强模式**: 可选 LLM 生成自然语言推理链

## 环境

```
Python 3.10+
CatBoost, Pandas, NumPy, scikit-learn, openpyxl, Flask
(可选) transformers, torch (BERT 对比), openai (LLM 评测/推理)
```