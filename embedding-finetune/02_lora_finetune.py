"""
Step 2: LoRA 微调 bge-large-zh-v1.5
使用 Step 1 生成的三元组数据，对 Embedding 模型进行 LoRA 微调。

特点：
- LoRA 只微调 ~0.5% 的参数，显存需求低（8GB GPU 即可）
- 使用 TripletLoss 训练，让相关的 query-document 对更近，不相关的更远
- 支持断点续训

使用：python 02_lora_finetune.py
"""

import json
import os
import logging
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class TrainingConfig:
    # 模型
    model_name: str = "BAAI/bge-large-zh-v1.5"
    
    # LoRA 参数
    lora_r: int = 16                    # 低秩矩阵的秩（8-64，越大表达能力越强但参数越多）
    lora_alpha: int = 32                # 缩放系数，通常设为 2*r
    lora_dropout: float = 0.1           # LoRA 层的 dropout
    lora_target_modules: list = field(default_factory=lambda: [
        "query", "value"                # 只微调注意力层的 Q 和 V 投影矩阵
    ])
    
    # 训练参数
    epochs: int = 5
    batch_size: int = 16                
    learning_rate: float = 2e-4         # LoRA 推荐 lr，比全量微调大
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 256           # Embedding 最大 token 长度
    margin: float = 0.3                 # TripletLoss 的 margin
    
    # 路径
    data_dir: str = "./data"
    output_dir: str = "./output/lora_checkpoint"
    log_interval: int = 50              # 每 N 步打印一次 loss
    save_interval: int = 500            # 每 N 步保存一次 checkpoint
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True                   # 混合精度训练（节省显存）


config = TrainingConfig()


# ============================================================
# 数据集
# ============================================================

class TripletDataset(Dataset):
    """三元组数据集：(query, positive, negative)"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"加载数据集: {data_path} ({len(self.data)} 条)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": item["negative"],
        }
    
    def collate_fn(self, batch):
        """自定义 batch 处理"""
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        negatives = [item["negative"] for item in batch]
        
        # 对 query 添加检索前缀（bge 模型的推荐做法）
        queries = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]
        
        q_encoded = self.tokenizer(
            queries, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        p_encoded = self.tokenizer(
            positives, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        n_encoded = self.tokenizer(
            negatives, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        
        return {
            "query": q_encoded,
            "positive": p_encoded,
            "negative": n_encoded,
        }


# ============================================================
# 模型
# ============================================================

def mean_pooling(model_output, attention_mask):
    """Mean Pooling：取所有 token embedding 的加权平均作为句子 embedding"""
    token_embeddings = model_output[0]  # (batch, seq_len, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode(model, tokenized_input, device):
    """编码文本为 embedding 向量"""
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = mean_pooling(outputs, attention_mask)
    
    # L2 归一化
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


class TripletLoss(torch.nn.Module):
    """三元组损失函数：让 query 离 positive 更近，离 negative 更远"""
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, q_emb, p_emb, n_emb):
        # 余弦相似度
        pos_sim = torch.nn.functional.cosine_similarity(q_emb, p_emb)
        neg_sim = torch.nn.functional.cosine_similarity(q_emb, n_emb)
        
        # TripletMarginLoss: max(0, neg_sim - pos_sim + margin)
        loss = torch.clamp(neg_sim - pos_sim + self.margin, min=0.0)
        return loss.mean()


# ============================================================
# 训练
# ============================================================

def setup_model(config: TrainingConfig):
    """加载模型并应用 LoRA"""
    
    logger.info(f"加载 base 模型: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    
    # 打印原始模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"原始模型参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
    
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # 打印 LoRA 参数统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA 可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    logger.info(f"参数占比: {100 * trainable_params / all_params:.2f}%")
    
    model.print_trainable_parameters()
    model.to(config.device)
    
    return model, tokenizer


def train(config: TrainingConfig):
    """主训练循环"""
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 1. 加载模型
    model, tokenizer = setup_model(config)
    
    # 2. 加载数据
    train_dataset = TripletDataset(
        os.path.join(config.data_dir, "triplets_train.json"),
        tokenizer, config.max_seq_length
    )
    val_dataset = TripletDataset(
        os.path.join(config.data_dir, "triplets_val.json"),
        tokenizer, config.max_seq_length
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=train_dataset.collate_fn,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=val_dataset.collate_fn,
        num_workers=0, pin_memory=True
    )
    
    # 3. 优化器 + 学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    loss_fn = TripletLoss(margin=config.margin)
    scaler = torch.amp.GradScaler("cuda") if config.fp16 and config.device == "cuda" else None
    
    logger.info(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")
    logger.info(f"设备: {config.device}, FP16: {config.fp16 and config.device == 'cuda'}")
    
    # 4. 训练循环
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        
        for batch in progress:
            optimizer.zero_grad()
            
            if config.fp16 and scaler is not None:
                with torch.amp.autocast("cuda"):
                    q_emb = encode(model, batch["query"], config.device)
                    p_emb = encode(model, batch["positive"], config.device)
                    n_emb = encode(model, batch["negative"], config.device)
                    loss = loss_fn(q_emb, p_emb, n_emb)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                q_emb = encode(model, batch["query"], config.device)
                p_emb = encode(model, batch["positive"], config.device)
                n_emb = encode(model, batch["negative"], config.device)
                loss = loss_fn(q_emb, p_emb, n_emb)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            
            # 进度条更新
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # 定期打印
            if global_step % config.log_interval == 0:
                avg_loss = sum(train_losses[-config.log_interval:]) / config.log_interval
                logger.info(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
            
            # 定期保存
            if global_step % config.save_interval == 0:
                save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(save_path)
                logger.info(f"保存 checkpoint: {save_path}")
        
        # Epoch 结束：验证
        avg_epoch_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn, config)
        
        logger.info(
            f"Epoch {epoch + 1} 完成 | "
            f"训练 Loss: {avg_epoch_loss:.4f} | "
            f"验证 Loss: {val_loss:.4f}"
        )
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.output_dir, "best_model")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info(f"新最优模型！验证 Loss: {val_loss:.4f}, 保存到: {best_path}")
    
    # 5. 保存最终模型
    final_path = os.path.join(config.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"训练完成！最终模型保存到: {final_path}")
    
    # 保存训练日志
    log_path = os.path.join(config.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {k: str(v) for k, v in config.__dict__.items()},
            "best_val_loss": best_val_loss,
            "total_steps": global_step,
            "train_losses": train_losses,
        }, f, indent=2)
    
    return model, tokenizer


@torch.no_grad()
def evaluate(model, val_loader, loss_fn, config):
    """验证集评估"""
    model.eval()
    total_loss = 0.0
    
    for batch in val_loader:
        q_emb = encode(model, batch["query"], config.device)
        p_emb = encode(model, batch["positive"], config.device)
        n_emb = encode(model, batch["negative"], config.device)
        loss = loss_fn(q_emb, p_emb, n_emb)
        total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader) if len(val_loader) > 0 else 0.0


# ============================================================
# 主入口
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("Step 2: LoRA 微调 Embedding 模型")
    logger.info("=" * 60)
    
    # 检查数据是否存在
    train_path = os.path.join(config.data_dir, "triplets_train.json")
    if not os.path.exists(train_path):
        logger.error(f"训练数据不存在: {train_path}")
        logger.error("请先运行: python 01_data_prepare.py")
        return
    
    # 检查 GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # 根据显存自动调整 batch_size
        if gpu_mem < 8:
            config.batch_size = 4
            logger.info(f"显存较小，自动调整 batch_size={config.batch_size}")
        elif gpu_mem < 12:
            config.batch_size = 8
        elif gpu_mem < 20:
            config.batch_size = 16
        else:
            config.batch_size = 32
    else:
        logger.warning("未检测到 GPU，将使用 CPU 训练（速度会很慢）")
        config.fp16 = False
        config.batch_size = 4
    
    # 开始训练
    train(config)
    
    logger.info("\n" + "=" * 60)
    logger.info("微调完成！下一步执行: python 03_merge_and_eval.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
