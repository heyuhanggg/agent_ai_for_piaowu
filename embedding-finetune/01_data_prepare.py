"""
Step 1: Jira 数据预处理
从 MongoDB 中提取 Bug/Incident/Support 类型的 Issue，
构造 (query, positive, negative) 三元组，用于 Embedding 模型 LoRA 微调。

使用前：
1. 安装 MongoDB
2. 导入数据: mongorestore --gzip --archive=../ThePublicJiraDataset/3.\ DataDump/mongodump-JiraReposAnon.archive
3. pip install pymongo pandas tqdm scikit-learn
"""

import json
import random
import re
import os
from collections import defaultdict
from typing import Optional

import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

# ============================================================
# 配置
# ============================================================

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "JiraReposAnon"

# 输出目录
OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bug/Incident 相关的 Issue 类型（从 jira_issuetype_thematic_analysis.json 提取）
BUG_ISSUE_TYPES = {
    "Bug", "Defect", "Outage", "Incident", "Issue",
    "Public Security Vulnerability", "Story Defect",
    "Production Issue", "Build Failure",
}

SUPPORT_ISSUE_TYPES = {
    "Support Request", "Support Ticket", "Problem", "Problem Ticket",
    "Question", "Questions", "Ask a question", "IT Help",
    "Clarification", "App Incident", "Infrastructure",
}

MAINTENANCE_ISSUE_TYPES = {
    "Technical Debt", "Performance", "Quality Risk",
    "Investigation",
}

# 合并所有运维相关 Issue 类型
OPS_RELEVANT_TYPES = BUG_ISSUE_TYPES | SUPPORT_ISSUE_TYPES | MAINTENANCE_ISSUE_TYPES

# 已解决状态
RESOLVED_STATUSES = {
    "Resolved", "Closed", "Done", "Fixed", "Completed",
    "Verified", "Released",
}

# 最终训练数据量控制
MAX_TRAIN_SAMPLES = 50000   # 最多用 5 万条训练
MAX_EVAL_SAMPLES = 2000     # 评估集 2000 条
MIN_TEXT_LENGTH = 30        # 正/负样本最短文本长度
MAX_TEXT_LENGTH = 512       # 截断长度（token 级别由模型处理，这里按字符粗控）

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# 文本清洗
# ============================================================

def clean_text(text: Optional[str]) -> str:
    """清洗 Jira Issue 文本"""
    if not text:
        return ""
    
    # 去除 Jira 标记语法
    text = re.sub(r'\{code[^}]*\}.*?\{code\}', ' [CODE_BLOCK] ', text, flags=re.DOTALL)
    text = re.sub(r'\{noformat\}.*?\{noformat\}', ' [CODE_BLOCK] ', text, flags=re.DOTALL)
    text = re.sub(r'\{quote\}.*?\{quote\}', '', text, flags=re.DOTALL)
    
    # 去除图片/附件引用
    text = re.sub(r'![\w\-\.]+\.(png|jpg|gif|bmp|svg)(\|[^!]*)?\!', '', text)
    
    # 去除 Jira 链接标记
    text = re.sub(r'\[([^\]]*)\|([^\]]*)\]', r'\1', text)
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    
    # 去除 @mentions
    text = re.sub(r'\[~[^\]]+\]', '', text)
    
    # 去除 Jira 格式标记
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # bold
    text = re.sub(r'_([^_]+)_', r'\1', text)    # italic
    text = re.sub(r'\+([^+]+)\+', r'\1', text)  # underline
    text = re.sub(r'-([^-]+)-', r'\1', text)     # strikethrough
    text = re.sub(r'\{color[^}]*\}(.*?)\{color\}', r'\1', text, flags=re.DOTALL)
    
    # 去除 h1-h6 标记
    text = re.sub(r'h[1-6]\.\s*', '', text)
    
    # 去除多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    
    return text


def extract_resolution_info(issue: dict) -> str:
    """从 Issue 中提取解决方案相关信息"""
    parts = []
    
    fields = issue.get("fields", {})
    
    # 1. Issue 描述
    description = clean_text(fields.get("description", ""))
    if description:
        parts.append(description)
    
    # 2. 评论（可能包含解决方案讨论）
    comment_data = fields.get("comment", {})
    comments = comment_data.get("comments", []) if isinstance(comment_data, dict) else []
    
    # 取最后几条评论（通常包含解决方案）
    for comment in comments[-3:]:
        body = clean_text(comment.get("body", ""))
        if body and len(body) > 20:
            parts.append(body)
    
    # 3. Resolution 描述
    resolution = fields.get("resolution", {})
    if isinstance(resolution, dict) and resolution.get("description"):
        parts.append(clean_text(resolution["description"]))
    
    return "\n\n".join(parts)


# ============================================================
# 数据提取
# ============================================================

def extract_issues_from_mongo() -> list[dict]:
    """从 MongoDB 提取所有运维相关 Issue"""
    
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    
    all_issues = []
    collection_names = db.list_collection_names()
    
    print(f"发现 {len(collection_names)} 个集合（Jira 仓库）")
    
    for coll_name in tqdm(collection_names, desc="扫描 Jira 仓库"):
        collection = db[coll_name]
        
        # 查询运维相关类型的 Issue
        query = {
            "fields.issuetype.name": {"$in": list(OPS_RELEVANT_TYPES)}
        }
        
        projection = {
            "key": 1,
            "fields.summary": 1,
            "fields.description": 1,
            "fields.issuetype.name": 1,
            "fields.status.name": 1,
            "fields.resolution": 1,
            "fields.comment.comments.body": 1,
            "fields.priority.name": 1,
            "fields.components.name": 1,
            "fields.labels": 1,
        }
        
        cursor = collection.find(query, projection)
        
        for doc in cursor:
            fields = doc.get("fields", {})
            summary = fields.get("summary", "")
            status = fields.get("status", {})
            status_name = status.get("name", "") if isinstance(status, dict) else ""
            issue_type = fields.get("issuetype", {})
            type_name = issue_type.get("name", "") if isinstance(issue_type, dict) else ""
            
            if not summary or len(summary) < 10:
                continue
            
            # 提取完整解决方案文本
            resolution_text = extract_resolution_info(doc)
            if len(resolution_text) < MIN_TEXT_LENGTH:
                continue
            
            # 提取组件和标签（用于后续负样本筛选）
            components = []
            for comp in fields.get("components", []):
                if isinstance(comp, dict) and comp.get("name"):
                    components.append(comp["name"])
            
            labels = fields.get("labels", []) or []
            
            priority = fields.get("priority", {})
            priority_name = priority.get("name", "") if isinstance(priority, dict) else ""
            
            all_issues.append({
                "key": doc.get("key", ""),
                "source": coll_name,
                "summary": clean_text(summary),
                "resolution_text": resolution_text[:MAX_TEXT_LENGTH * 3],
                "type": type_name,
                "status": status_name,
                "is_resolved": status_name in RESOLVED_STATUSES,
                "priority": priority_name,
                "components": components,
                "labels": labels,
            })
    
    client.close()
    
    print(f"\n提取完成：共 {len(all_issues)} 条运维相关 Issue")
    print(f"  其中已解决: {sum(1 for i in all_issues if i['is_resolved'])} 条")
    
    # 按 Issue 类型统计
    type_counts = defaultdict(int)
    for issue in all_issues:
        type_counts[issue["type"]] += 1
    print("\nIssue 类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {t}: {c}")
    
    # 按来源统计
    source_counts = defaultdict(int)
    for issue in all_issues:
        source_counts[issue["source"]] += 1
    print("\n来源分布:")
    for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")
    
    return all_issues


# ============================================================
# 加载本地运维知识库（用于混合训练）
# ============================================================

def load_local_ops_knowledge() -> list[dict]:
    """加载本地运维知识库 Markdown，构造补充训练样本"""
    
    ops_knowledge_dir = os.path.join(
        os.path.dirname(__file__), "..",
        "damai-core-service", "src", "main", "resources", "ops-knowledge"
    )
    
    local_pairs = []
    
    if not os.path.exists(ops_knowledge_dir):
        print(f"本地运维知识库目录不存在: {ops_knowledge_dir}")
        return local_pairs
    
    # 手动构造 query-document 对
    # 从每个 Markdown 的章节标题作为 query，章节内容作为 positive
    for filename in os.listdir(ops_knowledge_dir):
        if not filename.endswith(".md"):
            continue
        
        filepath = os.path.join(ops_knowledge_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 按 "---" 分割为独立章节
        sections = content.split("---")
        
        for section in sections:
            section = section.strip()
            if len(section) < 50:
                continue
            
            # 提取章节标题（## 开头的行）
            lines = section.split("\n")
            title = ""
            body = ""
            for line in lines:
                if line.startswith("## ") and not title:
                    title = line.replace("## ", "").strip()
                elif line.startswith("### ") and not title:
                    title = line.replace("### ", "").strip()
            
            body = section
            
            if not title or len(body) < 50:
                continue
            
            # 构造多种 query 变体
            queries = generate_query_variants(title, body)
            
            for query in queries:
                local_pairs.append({
                    "key": f"local_{filename}",
                    "source": "damai_ops_knowledge",
                    "summary": query,
                    "resolution_text": body[:MAX_TEXT_LENGTH * 2],
                    "type": "OpsKnowledge",
                    "status": "Active",
                    "is_resolved": True,
                    "priority": "High",
                    "components": [],
                    "labels": [],
                })
    
    print(f"本地运维知识库：构造了 {len(local_pairs)} 个训练样本")
    return local_pairs


def generate_query_variants(title: str, body: str) -> list[str]:
    """为一个知识库章节生成多种查询变体"""
    variants = [title]
    
    # 基于标题生成口语化查询
    query_templates = [
        f"{title}怎么办",
        f"{title}怎么排查",
        f"{title}怎么处理",
        f"{title}怎么解决",
        f"遇到{title}",
    ]
    
    # 根据内容中的关键词生成更具体的查询
    if "OOM" in body or "OutOfMemoryError" in body or "内存溢出" in body:
        variants.extend(["服务OOM了怎么办", "Java堆内存溢出", "OutOfMemoryError排查"])
    if "连接池" in body or "connection pool" in body.lower():
        variants.extend(["数据库连接池满了", "连接池耗尽怎么排查", "HikariPool连接超时"])
    if "Redis" in body:
        variants.extend(["Redis连接失败", "缓存异常怎么排查"])
    if "超时" in body or "timeout" in body.lower():
        variants.extend(["接口响应超时", "请求超时怎么排查"])
    if "慢SQL" in body or "慢查询" in body:
        variants.extend(["数据库慢查询", "SQL执行太慢"])
    if "发布" in body or "回滚" in body:
        variants.extend(["发布后出现问题需要回滚", "服务发布流程"])
    if "扩容" in body:
        variants.extend(["服务需要扩容", "CPU过高需要扩容"])
    if "缓存雪崩" in body:
        variants.extend(["缓存全部失效了", "大量请求穿透到数据库"])
    
    variants.extend(query_templates)
    
    # 去重
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    
    return unique


# ============================================================
# 三元组构造
# ============================================================

def build_triplets(issues: list[dict]) -> list[dict]:
    """构造 (query, positive, negative) 三元组"""
    
    print(f"\n开始构造训练三元组（总样本: {len(issues)}）...")
    
    # 按来源分组，用于 hard negative 采样
    source_groups = defaultdict(list)
    for idx, issue in enumerate(issues):
        source_groups[issue["source"]].append(idx)
    
    # 按 Issue 类型分组
    type_groups = defaultdict(list)
    for idx, issue in enumerate(issues):
        type_groups[issue["type"]].append(idx)
    
    triplets = []
    
    for i, issue in enumerate(tqdm(issues, desc="构造三元组")):
        query = issue["summary"]
        positive = issue["resolution_text"]
        
        if len(query) < 10 or len(positive) < MIN_TEXT_LENGTH:
            continue
        
        # 截断
        positive = positive[:MAX_TEXT_LENGTH * 2]
        
        # === 负样本采样策略 ===
        # 策略1: 同来源不同类型的 Issue（hard negative - 同项目但不同问题）
        # 策略2: 不同来源的 Issue（easy negative - 完全不同的项目）
        # 策略3: 同类型不同来源的 Issue（medium negative - 同类型但不同项目）
        
        negatives = []
        
        # Hard negative: 同来源、不同 Issue
        same_source = source_groups.get(issue["source"], [])
        if len(same_source) > 1:
            candidates = [j for j in same_source if j != i]
            if candidates:
                neg_idx = random.choice(candidates)
                neg_text = issues[neg_idx]["resolution_text"][:MAX_TEXT_LENGTH * 2]
                if len(neg_text) >= MIN_TEXT_LENGTH:
                    negatives.append(neg_text)
        
        # Easy negative: 不同来源
        other_sources = [s for s in source_groups.keys() if s != issue["source"]]
        if other_sources:
            neg_source = random.choice(other_sources)
            neg_idx = random.choice(source_groups[neg_source])
            neg_text = issues[neg_idx]["resolution_text"][:MAX_TEXT_LENGTH * 2]
            if len(neg_text) >= MIN_TEXT_LENGTH:
                negatives.append(neg_text)
        
        if not negatives:
            continue
        
        # 每个 query 生成 1-2 个三元组
        for negative in negatives[:2]:
            triplets.append({
                "query": query,
                "positive": positive,
                "negative": negative,
                "source": issue["source"],
                "type": issue["type"],
            })
    
    print(f"构造完成：共 {len(triplets)} 个三元组")
    return triplets


# ============================================================
# 数据集划分与保存
# ============================================================

def split_and_save(triplets: list[dict]):
    """划分训练集/验证集/测试集并保存"""
    
    random.shuffle(triplets)
    
    # 控制总量
    if len(triplets) > MAX_TRAIN_SAMPLES + MAX_EVAL_SAMPLES:
        triplets = triplets[:MAX_TRAIN_SAMPLES + MAX_EVAL_SAMPLES]
    
    # 90% 训练，5% 验证，5% 测试
    n = len(triplets)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)
    
    train_data = triplets[:train_end]
    val_data = triplets[train_end:val_end]
    test_data = triplets[val_end:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")
    print(f"  测试集: {len(test_data)}")
    
    # 保存为 JSON
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        filepath = os.path.join(OUTPUT_DIR, f"triplets_{name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  保存: {filepath} ({len(data)} 条)")
    
    # 同时保存为 sentence-transformers 兼容的 CSV 格式
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        rows = []
        for item in data:
            rows.append({
                "anchor": item["query"],
                "positive": item["positive"],
                "negative": item["negative"],
            })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUTPUT_DIR, f"triplets_{name}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  保存 CSV: {csv_path}")
    
    # 保存统计信息
    stats = {
        "total_triplets": len(triplets),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "avg_query_length": sum(len(t["query"]) for t in triplets) / len(triplets),
        "avg_positive_length": sum(len(t["positive"]) for t in triplets) / len(triplets),
        "avg_negative_length": sum(len(t["negative"]) for t in triplets) / len(triplets),
        "source_distribution": dict(pd.Series([t["source"] for t in triplets]).value_counts()),
        "type_distribution": dict(pd.Series([t["type"] for t in triplets]).value_counts()),
    }
    
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n统计信息: {stats_path}")
    
    return train_data, val_data, test_data


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("Step 1: Jira 数据预处理 → 训练三元组")
    print("=" * 60)
    
    # 1. 从 MongoDB 提取 Issue
    print("\n[1/4] 从 MongoDB 提取运维相关 Issue...")
    issues = extract_issues_from_mongo()
    
    # 2. 加载本地运维知识库
    print("\n[2/4] 加载本地运维知识库...")
    local_issues = load_local_ops_knowledge()
    
    # 合并（本地数据重复加入以提高权重）
    all_issues = issues + local_issues * 3  # 本地知识库 3 倍采样
    print(f"\n合并后总样本: {len(all_issues)} (Jira: {len(issues)}, 本地x3: {len(local_issues) * 3})")
    
    # 3. 构造三元组
    print("\n[3/4] 构造训练三元组...")
    triplets = build_triplets(all_issues)
    
    # 4. 划分并保存
    print("\n[4/4] 划分数据集并保存...")
    split_and_save(triplets)
    
    print("\n" + "=" * 60)
    print("数据预处理完成！下一步执行: python 02_lora_finetune.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
