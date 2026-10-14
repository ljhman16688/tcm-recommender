FROM python:3.12-slim

# 环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV LLM_API_KEY=""
ENV LLM_BASE_URL="https://api.zhizengzeng.com/v1"
ENV LLM_MODEL="qwen3.7-max"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 只复制 Web 推理所需文件，避免把论文、实验脚本和原始数据放入镜像。
# 推理核改为 712 冻结口径（src/deploy_712 + protocol_712 + kg_pmi_features），需复制整个 src/。
COPY web_app/ ./web_app/
COPY src/ ./src/
COPY data/herb_property_dict_merged.json data/tcm_knowledge_graph.json data/dosage_form_rules.json ./data/
COPY data/deploy/pmic_agg_712.json ./data/deploy/
COPY gunicorn.conf.py .

# 部署使用冻结脚本 scripts/utils/freeze_deploy_assets.py 产出的模型与聚合统计，
# 不在容器启动时重新训练（与论文主模型 G 一致，测试集 Top-1=45.15%）。
RUN test -s web_app/catboost_G.cbm && \
    test -s data/deploy/pmic_agg_712.json && \
    test -s data/herb_property_dict_merged.json

# 创建 outputs 目录（运行时写入日志）
RUN mkdir -p /app/outputs

EXPOSE 5000

# Gunicorn 生产服务器
CMD ["gunicorn", "--config", "gunicorn.conf.py", "web_app.app:app"]
