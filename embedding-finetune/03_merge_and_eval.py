"""
Step 3: 合并 LoRA 权重 + 检索效果评估
将 LoRA adapter 权重合并回 base 模型，生成一个独立可部署的完整模型，
并在测试集上评估检索效果（MRR、Hit@K、相似度分布）。

使用：python 03_merge_and_eval.py
"""

import json
import os
import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

BASE_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
LORA_CHECKPOINT = "./output/lora_checkpoint/best_model"   # LoRA adapter 路径
MERGED_OUTPUT_DIR = "./output/merged_model"                # 合并后完整模型路径
TEST_DATA_PATH = "./data/triplets_test.json"
EVAL_RESULTS_PATH = "./output/eval_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32


# ============================================================
# 合并 LoRA 权重
# ============================================================

def merge_lora_weights():
    """将 LoRA adapter 合并回 base 模型"""
    
    logger.info("=" * 50)
    logger.info("合并 LoRA 权重到 base 模型")
    logger.info("=" * 50)
    
    # 1. 加载 base 模型
    logger.info(f"加载 base 模型: {BASE_MODEL_NAME}")
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # 2. 加载 LoRA adapter
    logger.info(f"加载 LoRA adapter: {LORA_CHECKPOINT}")
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
    
    # 3. 合并权重
    logger.info("合并 LoRA 权重...")
    merged_model = model.merge_and_unload()
    
    # 4. 保存合并后的完整模型
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    
    total_params = sum(p.numel() for p in merged_model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in merged_model.parameters()) / 1024**2
    
    logger.info(f"合并完成！")
    logger.info(f"  模型参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
    logger.info(f"  模型大小: {model_size_mb:.1f} MB")
    logger.info(f"  保存路径: {MERGED_OUTPUT_DIR}")
    
    return merged_model, tokenizer


# ============================================================
# Embedding 编码
# ============================================================

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], is_query: bool = False,
                 batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """批量编码文本为 embedding"""
    model.eval()
    all_embeddings = []
    
    # bge 模型对 query 添加前缀
    if is_query:
        texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=MAX_SEQ_LENGTH, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        outputs = model(**encoded)
        embeddings = mean_pooling(outputs, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


# ============================================================
# 评估
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算余弦相似度"""
    return np.sum(a * b, axis=-1)


def evaluate_retrieval(model, tokenizer, test_data: list[dict], device: str):
    """评估检索效果"""
    
    logger.info(f"评估数据量: {len(test_data)} 条")
    
    queries = [item["query"] for item in test_data]
    positives = [item["positive"] for item in test_data]
    negatives = [item["negative"] for item in test_data]
    
    logger.info("编码 queries...")
    q_embeddings = encode_texts(model, tokenizer, queries, is_query=True,
                                batch_size=BATCH_SIZE, device=device)
    
    logger.info("编码 positives...")
    p_embeddings = encode_texts(model, tokenizer, positives, is_query=False,
                                batch_size=BATCH_SIZE, device=device)
    
    logger.info("编码 negatives...")
    n_embeddings = encode_texts(model, tokenizer, negatives, is_query=False,
                                batch_size=BATCH_SIZE, device=device)
    
    # 计算相似度
    pos_similarities = cosine_similarity(q_embeddings, p_embeddings)
    neg_similarities = cosine_similarity(q_embeddings, n_embeddings)
    
    # === 指标计算 ===
    
    # 1. 正样本相似度 > 负样本相似度的比例（准确率）
    correct = (pos_similarities > neg_similarities).sum()
    accuracy = correct / len(test_data)
    
    # 2. 平均正样本相似度
    avg_pos_sim = pos_similarities.mean()
    
    # 3. 平均负样本相似度
    avg_neg_sim = neg_similarities.mean()
    
    # 4. 正负样本相似度差距
    sim_gap = (pos_similarities - neg_similarities).mean()
    
    # 5. MRR (Mean Reciprocal Rank) - 在 positive + negative 中排名
    mrr_scores = []
    for i in range(len(test_data)):
        scores = [(pos_similarities[i], True), (neg_similarities[i], False)]
        scores.sort(key=lambda x: -x[0])
        for rank, (score, is_pos) in enumerate(scores, 1):
            if is_pos:
                mrr_scores.append(1.0 / rank)
                break
    mrr = np.mean(mrr_scores)
    
    # 6. Hit@1 - positive 是否排名第一
    hit_at_1 = accuracy  # 在二选一的场景下，Hit@1 等于准确率
    
    # 7. 大规模检索模拟：每个 query 在所有 positive+negative 中检索
    logger.info("大规模检索评估（每个 query 在全部文档中检索）...")
    all_doc_embeddings = np.concatenate([p_embeddings, n_embeddings], axis=0)
    
    hit_at_5 = 0
    hit_at_10 = 0
    mrr_full = []
    
    for i in tqdm(range(len(queries)), desc="检索评估"):
        # 计算 query 与所有文档的相似度
        sims = cosine_similarity(
            q_embeddings[i:i+1].repeat(len(all_doc_embeddings), axis=0),
            all_doc_embeddings
        )
        # 正确文档的索引是 i（在 p_embeddings 中的位置）
        correct_idx = i
        
        # 排名
        sorted_indices = np.argsort(-sims)
        rank = np.where(sorted_indices == correct_idx)[0][0] + 1
        
        mrr_full.append(1.0 / rank)
        if rank <= 5:
            hit_at_5 += 1
        if rank <= 10:
            hit_at_10 += 1
    
    n = len(queries)
    
    results = {
        "dataset_size": len(test_data),
        "pairwise_accuracy": float(accuracy),
        "avg_positive_similarity": float(avg_pos_sim),
        "avg_negative_similarity": float(avg_neg_sim),
        "similarity_gap": float(sim_gap),
        "pairwise_mrr": float(mrr),
        "pairwise_hit_at_1": float(hit_at_1),
        "fullset_mrr": float(np.mean(mrr_full)),
        "fullset_hit_at_5": float(hit_at_5 / n),
        "fullset_hit_at_10": float(hit_at_10 / n),
        "pos_sim_distribution": {
            "min": float(pos_similarities.min()),
            "max": float(pos_similarities.max()),
            "mean": float(pos_similarities.mean()),
            "std": float(pos_similarities.std()),
            "p25": float(np.percentile(pos_similarities, 25)),
            "p50": float(np.percentile(pos_similarities, 50)),
            "p75": float(np.percentile(pos_similarities, 75)),
        },
        "neg_sim_distribution": {
            "min": float(neg_similarities.min()),
            "max": float(neg_similarities.max()),
            "mean": float(neg_similarities.mean()),
            "std": float(neg_similarities.std()),
            "p25": float(np.percentile(neg_similarities, 25)),
            "p50": float(np.percentile(neg_similarities, 50)),
            "p75": float(np.percentile(neg_similarities, 75)),
        },
    }
    
    return results


def compare_base_vs_finetuned(test_data: list[dict]):
    """对比 base 模型和微调模型的检索效果"""
    
    logger.info("\n" + "=" * 50)
    logger.info("对比评估：Base 模型 vs LoRA 微调模型")
    logger.info("=" * 50)
    
    all_results = {}
    
    # 1. 评估 base 模型
    logger.info("\n--- 评估 Base 模型 ---")
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME).to(DEVICE)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_results = evaluate_retrieval(base_model, base_tokenizer, test_data, DEVICE)
    all_results["base_model"] = base_results
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 2. 评估微调模型
    logger.info("\n--- 评估 LoRA 微调模型 ---")
    ft_model = AutoModel.from_pretrained(MERGED_OUTPUT_DIR).to(DEVICE)
    ft_tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT_DIR)
    ft_results = evaluate_retrieval(ft_model, ft_tokenizer, test_data, DEVICE)
    all_results["finetuned_model"] = ft_results
    del ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 3. 打印对比表
    logger.info("\n" + "=" * 60)
    logger.info("评估结果对比")
    logger.info("=" * 60)
    
    metrics = [
        ("配对准确率", "pairwise_accuracy"),
        ("配对 MRR", "pairwise_mrr"),
        ("全集 MRR", "fullset_mrr"),
        ("全集 Hit@5", "fullset_hit_at_5"),
        ("全集 Hit@10", "fullset_hit_at_10"),
        ("正样本平均相似度", "avg_positive_similarity"),
        ("负样本平均相似度", "avg_negative_similarity"),
        ("正负相似度差距", "similarity_gap"),
    ]
    
    logger.info(f"{'指标':<20} {'Base 模型':>12} {'微调模型':>12} {'提升':>10}")
    logger.info("-" * 56)
    for name, key in metrics:
        base_val = base_results[key]
        ft_val = ft_results[key]
        diff = ft_val - base_val
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        logger.info(f"{name:<20} {base_val:>12.4f} {ft_val:>12.4f} {arrow}{abs(diff):>8.4f}")
    
    return all_results


# ============================================================
# 运维术语专项测试
# ============================================================

def ops_terminology_test(model, tokenizer):
    """运维术语语义理解专项测试"""
    
    logger.info("\n" + "=" * 50)
    logger.info("运维术语语义理解专项测试")
    logger.info("=" * 50)
    
    # 测试用例：(query, 应该相似的文本, 不应该相似的文本)
    test_cases = [
        {
            "query": "OOM",
            "should_match": ["OutOfMemoryError", "内存溢出", "堆内存耗尽", "Java heap space"],
            "should_not_match": ["订单创建", "用户登录", "节目推荐"],
        },
        {
            "query": "连接池耗尽",
            "should_match": ["connection pool exhausted", "HikariPool timeout", "数据库连接不够用"],
            "should_not_match": ["游泳池", "内存不足", "磁盘空间"],
        },
        {
            "query": "缓存雪崩",
            "should_match": ["cache avalanche", "缓存同时过期", "大量请求穿透到数据库"],
            "should_not_match": ["下雪了", "缓存命中率", "数据备份"],
        },
        {
            "query": "慢SQL",
            "should_match": ["slow query", "全表扫描", "缺少索引", "查询超时"],
            "should_not_match": ["快速排序", "SQL注入", "数据迁移"],
        },
        {
            "query": "服务发布回滚",
            "should_match": ["rollback deployment", "版本回退", "发布失败需要回滚"],
            "should_not_match": ["服务注册", "用户反馈", "日志查询"],
        },
        {
            "query": "Full GC频繁",
            "should_match": ["频繁垃圾回收", "GC暂停时间长", "STW导致服务卡顿"],
            "should_not_match": ["磁盘清理", "日志轮转", "版本发布"],
        },
    ]
    
    results = []
    
    for case in test_cases:
        query = case["query"]
        all_candidates = case["should_match"] + case["should_not_match"]
        
        q_emb = encode_texts(model, tokenizer, [query], is_query=True, device=DEVICE)
        c_embs = encode_texts(model, tokenizer, all_candidates, is_query=False, device=DEVICE)
        
        similarities = cosine_similarity(
            q_emb.repeat(len(all_candidates), axis=0), c_embs
        )
        
        logger.info(f"\nQuery: \"{query}\"")
        
        # 按相似度排序
        sorted_pairs = sorted(zip(all_candidates, similarities), key=lambda x: -x[1])
        
        correct_top = 0
        n_should_match = len(case["should_match"])
        
        for rank, (text, sim) in enumerate(sorted_pairs, 1):
            is_match = text in case["should_match"]
            marker = "✓" if is_match else "✗"
            logger.info(f"  #{rank} [{marker}] sim={sim:.4f} \"{text}\"")
            if rank <= n_should_match and is_match:
                correct_top += 1
        
        precision = correct_top / n_should_match
        results.append({
            "query": query,
            "precision_at_k": precision,
            "details": sorted_pairs,
        })
    
    avg_precision = np.mean([r["precision_at_k"] for r in results])
    logger.info(f"\n运维术语测试平均 Precision@K: {avg_precision:.4f}")
    
    return results


# ============================================================
# 主入口
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("Step 3: 合并 LoRA 权重 + 评估检索效果")
    logger.info("=" * 60)
    
    # 1. 合并 LoRA
    if not os.path.exists(LORA_CHECKPOINT):
        logger.error(f"LoRA checkpoint 不存在: {LORA_CHECKPOINT}")
        logger.error("请先运行: python 02_lora_finetune.py")
        return
    
    merged_model, tokenizer = merge_lora_weights()
    
    # 2. 加载测试数据
    if not os.path.exists(TEST_DATA_PATH):
        logger.warning(f"测试数据不存在: {TEST_DATA_PATH}，跳过量化评估")
    else:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        # 3. 对比评估
        all_results = compare_base_vs_finetuned(test_data)
        
        # 保存评估结果
        os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
        with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n评估结果已保存: {EVAL_RESULTS_PATH}")
    
    # 4. 运维术语专项测试
    merged_model = merged_model.to(DEVICE)
    ops_terminology_test(merged_model, tokenizer)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"合并后模型路径: {MERGED_OUTPUT_DIR}")
    logger.info("下一步执行: python 04_deploy_server.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
