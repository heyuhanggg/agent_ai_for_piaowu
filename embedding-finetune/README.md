# Embedding 模型 LoRA 微调工具集

基于 ThePublicJiraDataset 数据集，对 `bge-large-zh-v1.5` 进行 LoRA 微调，
提升运维 RAG 检索中对技术术语的语义理解能力。

## 文件说明

| 文件 | 说明 |
|------|------|
| `01_data_prepare.py` | Jira MongoDB 数据导入 + 筛选 Bug/Incident + 三元组构造 |
| `02_lora_finetune.py` | LoRA 微调 bge-large-zh-v1.5 训练脚本 |
| `03_merge_and_eval.py` | 合并 LoRA 权重 + 检索效果评估 |
| `04_deploy_server.py` | FastAPI 部署为 OpenAI 兼容 Embedding API |
| `requirements.txt` | Python 依赖 |

## 使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 导入 Jira 数据到 MongoDB（需先安装 MongoDB）
mongorestore --gzip --archive=../ThePublicJiraDataset/3.\ DataDump/mongodump-JiraReposAnon.archive

# 3. 数据预处理：提取训练三元组
python 01_data_prepare.py

# 4. LoRA 微调（建议 GPU，至少 8GB 显存）
python 02_lora_finetune.py

# 5. 合并权重 + 评估效果
python 03_merge_and_eval.py

# 6. 部署微调后的模型
python 04_deploy_server.py
```

## 硬件要求

| 阶段 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 数据预处理 | 16GB RAM | 32GB RAM |
| LoRA 微调 | 8GB VRAM GPU | 16GB+ VRAM GPU (A100/4090) |
| 模型部署 | 4GB VRAM GPU | 8GB+ VRAM GPU |

> 如果没有 GPU，可以使用 Google Colab (免费 T4 GPU) 进行训练。
