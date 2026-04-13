"""
Step 4: 部署微调后的 Embedding 模型
使用 FastAPI 封装为 OpenAI 兼容的 /v1/embeddings API，
可以直接替换 Spring AI 中的 embedding 配置。

使用：python 04_deploy_server.py
访问：http://localhost:8100/v1/embeddings
文档：http://localhost:8100/docs
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "./output/merged_model")
HOST = os.environ.get("EMBEDDING_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMBEDDING_PORT", "8100"))
MAX_SEQ_LENGTH = int(os.environ.get("EMBEDDING_MAX_SEQ_LENGTH", "256"))
MAX_BATCH_SIZE = int(os.environ.get("EMBEDDING_MAX_BATCH_SIZE", "64"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# bge 模型的查询前缀
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


# ============================================================
# 模型加载
# ============================================================

class EmbeddingModelWrapper:
    """Embedding 模型封装"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        logger.info(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型参数量: {total_params:,} ({total_params / 1e6:.1f}M)")
        logger.info(f"设备: {device}")
        
        # 预热
        self._warmup()
    
    def _warmup(self):
        """预热模型，避免首次请求延迟"""
        logger.info("模型预热中...")
        _ = self.encode(["warmup text"])
        logger.info("模型预热完成")
    
    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """编码文本列表为 embedding 向量"""
        
        if is_query:
            texts = [f"{QUERY_PREFIX}{t}" for t in texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]
            
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_SEQ_LENGTH, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            outputs = self.model(**encoded)
            
            # Mean Pooling
            token_embeddings = outputs[0]
            attention_mask = encoded["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
            
            # L2 归一化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().numpy().tolist())
        
        return all_embeddings


# ============================================================
# API 定义（OpenAI 兼容格式）
# ============================================================

app = FastAPI(
    title="Ops Embedding API",
    description="LoRA 微调后的运维领域 Embedding 模型，OpenAI 兼容接口",
    version="1.0.0",
)

# 全局模型实例
model_wrapper: Optional[EmbeddingModelWrapper] = None


class EmbeddingRequest(BaseModel):
    """OpenAI 兼容的 embedding 请求格式"""
    input: str | list[str] = Field(..., description="要编码的文本，支持单个字符串或字符串列表")
    model: str = Field(default="ops-embedding", description="模型名称")
    encoding_format: str = Field(default="float", description="编码格式")
    is_query: bool = Field(default=False, description="是否为查询文本（会添加检索前缀）")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI 兼容的 embedding 响应格式"""
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "damai-ops"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ============================================================
# API 路由
# ============================================================

@app.on_event("startup")
async def startup():
    """服务启动时加载模型"""
    global model_wrapper
    model_wrapper = EmbeddingModelWrapper(MODEL_PATH, DEVICE)
    logger.info(f"Embedding API 已启动: http://{HOST}:{PORT}")
    logger.info(f"API 文档: http://{HOST}:{PORT}/docs")


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """
    创建文本的 embedding 向量（OpenAI 兼容接口）
    
    Spring AI 配置示例：
    ```yaml
    spring:
      ai:
        openai:
          embedding:
            base-url: http://localhost:8100
            api-key: not-needed
            options:
              model: ops-embedding
    ```
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    
    # 统一转为列表
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input
    
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="input 不能为空")
    
    if len(texts) > 256:
        raise HTTPException(status_code=400, detail="单次请求最多 256 条文本")
    
    start_time = time.time()
    
    # 编码
    embeddings = model_wrapper.encode(texts, is_query=request.is_query)
    
    elapsed = time.time() - start_time
    logger.info(f"编码 {len(texts)} 条文本，耗时 {elapsed:.3f}s")
    
    # 估算 token 数
    total_tokens = sum(len(t) for t in texts)  # 粗略估算
    
    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ],
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """列出可用模型"""
    return ModelList(data=[
        ModelInfo(id="ops-embedding"),
    ])


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model_wrapper is not None,
        "device": DEVICE,
        "model_path": MODEL_PATH,
    }


@app.get("/")
async def root():
    return {
        "service": "Ops Embedding API",
        "version": "1.0.0",
        "docs": f"http://{HOST}:{PORT}/docs",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "health": "/health",
        }
    }


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Step 4: 部署 Embedding 模型 API")
    logger.info("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"模型路径不存在: {MODEL_PATH}")
        logger.error("请先运行: python 03_merge_and_eval.py")
        exit(1)
    
    logger.info(f"模型路径: {MODEL_PATH}")
    logger.info(f"监听地址: {HOST}:{PORT}")
    logger.info(f"设备: {DEVICE}")
    
    uvicorn.run(
        "04_deploy_server:app",
        host=HOST,
        port=PORT,
        workers=1,       # Embedding 模型单 worker 即可
        log_level="info",
    )
