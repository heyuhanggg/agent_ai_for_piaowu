# 演出票务AI - 智能票务AI助手

基于 Spring AI 构建的智能票务系统，集成了多种AI模型和先进的RAG技术，为演出票务提供智能化服务。

## 项目特点

- 🤖 **多模型支持**: 集成 DeepSeek、Qwen Max、OpenAI 等多种大语言模型
- 📚 **RAG检索增强**: 实现混合检索（向量+关键词）+ Rerank精排的RAG方案
- 🔧 **MCP工具集成**: 支持运维日志查询、链路追踪等MCP工具调用
- 📊 **AI可观测性**: 完整的Token统计、成本分析、请求追踪能力
- 🎯 **Function Calling**: 支持节目推荐、订单创建等业务函数调用
- 💬 **多角色对话**: 贴心助手、规则助手、运维助手三种AI角色

## 技术架构

### 核心技术栈
- **框架**: Spring Boot 3.x + Spring AI
- **AI模型**: DeepSeek Chat, Qwen Max, OpenAI Embedding
- **数据库**: MySQL (会话历史), Elasticsearch (全文检索)
- **向量存储**: SimpleVectorStore
- **前端**: Vue 3 + TypeScript + Vite

### 系统架构
```
├── ticket-core-service      # 核心AI服务
│   ├── advisor/            # 自定义Advisor (可观测性、查询改写等)
│   ├── ai/                 # AI相关实现 (RAG、Function Calling)
│   ├── config/             # 配置类 (模型配置、MCP配置)
│   ├── service/            # 业务服务 (混合检索、Rerank)
│   └── controller/         # REST API
├── ticket-mcp-server        # MCP工具服务
│   ├── ticket-mcp-log-service      # 日志查询服务
│   └── ticket-mcp-metrics-service  # 指标监控服务
└── vue/                    # 前端界面
```

## 功能特性

### 1. 贴心助手 - ReAct 决策模式 🔄
**采用 ReAct (Reason + Act) 决策机制，在推理、工具调用与结果观察之间迭代执行**

- 🎫 **节目推荐**: 基于用户位置和偏好推荐演出
- 📝 **详情查询**: 提供节目详情、票价、演出时间
- 🛒 **智能下单**: 通过对话完成购票流程
- 💬 **多轮对话**: 保持上下文的自然交互
- 🔄 **ReAct循环**: Thought → Action → Observation，最多5轮迭代
- 🧠 **思考过程**: 每步都会先分析当前状态再决定行动

**工作流程：**
```
用户请求 → 思考(分析需求) → 行动(调用工具) → 观察(分析结果) → 继续思考...
```

### 2. 规则助手 (Qwen Max + RAG)
- 📚 **知识库问答**: 基于Markdown文档的规则查询
- 🔍 **混合检索**: 向量检索 + 关键词检索 + RRF融合
- 🎯 **Rerank精排**: 提升检索结果准确性
- 🔄 **查询改写**: 优化用户查询提升召回率

### 3. 运维助手 - Plan-Execute-Replan 模式 📋
**采用 Plan-Execute-Replan 模式，结合MCP工具实现多步诊断与动态重规划**

- 📊 **日志查询**: 通过MCP工具查询系统日志
- 🔗 **链路追踪**: 完整的请求链路分析
- 📈 **指标监控**: JVM内存、GC、线程状态等
- 🛠️ **智能诊断**: AI辅助问题定位和解决方案推荐
- 📋 **计划制定**: 自动制定详细的诊断步骤计划
- 🔄 **动态调整**: 根据执行结果动态调整后续计划，最多重规划3次

**工作流程：**
```
问题分析 → 制定计划 → 执行步骤 → 观察结果 → 重新规划(如需) → 最终结论
```

### 4. AI可观测性
- 💰 **成本统计**: Token用量和API调用费用
- ⏱️ **性能监控**: 请求延迟、响应时间
- 📉 **趋势分析**: 按时间、类型的统计图表
- 🔍 **调用追踪**: 完整的AI调用链路记录

## 快速开始

### 环境要求
- JDK 17+
- Maven 3.6+
- MySQL 8.0+
- Elasticsearch 8.x (可选)

### 配置说明

详细配置请查看 [CONFIG.md](CONFIG.md)

1. **复制配置模板**
```bash
cp ticket-core-service/src/main/resources/application.yaml.example \
   ticket-core-service/src/main/resources/application.yaml
```

2. **配置API密钥**
```yaml
spring:
  ai:
    openai:
      api-key: ${alibaba-key}  # 阿里云百炼API Key
    deepseek:
      api-key: ${deepseek-key}  # DeepSeek API Key
```

3. **配置数据库**
```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/ticket_ai
    username: root
    password: YOUR_PASSWORD
```

### 启动项目

```bash
# 1. 编译打包
mvn clean package -DskipTests

# 2. 启动后端
java -jar ticket-core-service/target/ticket-core-service-*.jar

# 3. 启动前端（可选）
cd vue
npm install
npm run dev
```

### API端点

- **贴心助手**: `POST /chat/assistant`
- **规则助手**: `POST /chat/markdown`
- **运维助手**: `POST /chat/analysis`
- **会话历史**: `GET /history/{conversationId}`
- **AI统计**: `GET /ai/observability/statistics`

## 核心实现

### RAG架构

#### V1版本 - QuestionAnswerAdvisor
```java
QuestionAnswerAdvisor.builder(vectorStore)
    .searchRequest(SearchRequest.builder()
        .similarityThreshold(0.3)
        .topK(8)
        .build())
    .build()
```

#### V2版本 - 混合检索 + Rerank
```java
// 1. 向量检索
List<Document> vectorResults = vectorStore.similaritySearch(query);

// 2. 关键词检索
List<Document> keywordResults = keywordSearch(query);

// 3. RRF融合
List<Document> merged = mergeWithRRF(vectorResults, keywordResults);

// 4. Rerank精排
List<Document> final = rerankService.rerank(query, merged, topK);
```

### Agent决策模式

#### ReAct (Reason + Act) 模式
购票助手采用的决策模式，让AI在思考和行动之间循环迭代：

```java
ReactAdvisor.builder()
    .maxIterations(5)           // 最多5轮迭代
    .enableReactLoop(true)      // 启用ReAct循环
    .order(ADVISOR_ORDER)
    .build()
```

**执行流程：**
1. **Thought (思考)**: 分析当前情况，思考下一步行动
2. **Action (行动)**: 调用工具执行具体操作
3. **Observation (观察)**: 观察工具返回结果
4. **Repeat**: 根据观察继续思考，直到完成任务

#### Plan-Execute-Replan 模式
运维助手采用的决策模式，制定计划并动态调整：

```java
PlanExecuteReplanAdvisor.builder()
    .maxReplans(3)              // 最多重规划3次
    .enablePlanning(true)       // 启用计划模式
    .order(ADVISOR_ORDER)
    .build()
```

**执行流程：**
1. **Plan (制定计划)**: 分析问题，制定诊断步骤（JSON格式）
2. **Execute (执行)**: 按步骤调用MCP工具，收集信息
3. **Observe (观察)**: 分析执行结果，评估是否达到目标
4. **Replan (重新规划)**: 根据新发现动态调整后续步骤

**计划示例：**
```json
{
  "goal": "诊断order-service错误",
  "steps": [
    {"id": 1, "action": "查询错误日志", "tool": "log_query", "status": "pending"},
    {"id": 2, "action": "追踪请求链路", "tool": "trace_query", "status": "pending"},
    {"id": 3, "action": "分析根因", "tool": "analyze", "status": "pending"}
  ]
}
```

### 自定义Advisor列表

- **ReactAdvisor**: ReAct决策模式，支持思考-行动-观察循环
- **PlanExecuteReplanAdvisor**: 计划-执行-重规划模式，支持动态调整
- **AiObservabilityAdvisor**: Token统计、成本计算、请求追踪
- **QueryRewriteAdvisor**: 查询改写优化
- **ChatTypeHistoryAdvisor**: 会话类型管理
- **ChatTypeTitleAdvisor**: 自动生成会话标题

### Function Calling示例

```java
@Description("推荐演出节目")
public ProgramRecommendation recommendPrograms(
    @Description("用户所在城市") String city,
    @Description("节目类型") String type
) {
    // 业务逻辑
    return new ProgramRecommendation(programs);
}
```

## 模型配置

| 角色 | 模型 | 决策模式 | 用途 |
|------|------|----------|------|
| 贴心助手 | DeepSeek Chat | **ReAct** | 通用对话、Function Calling、迭代式推理 |
| 规则助手 | Qwen Max | 标准RAG | RAG知识问答、混合检索 |
| 运维助手 | DeepSeek Chat | **Plan-Execute-Replan** | MCP工具调用、多步诊断、动态规划 |
| 标题生成 | DeepSeek Chat | 简单生成 | 会话标题自动生成 |
| Embedding | text-embedding-v3 | - | 向量化 (1024维) |

## 项目结构

```
ticket-ai/
├── ticket-core-service/           # 核心服务
│   ├── src/main/java/
│   │   ├── advisor/              # 自定义Advisor
│   │   ├── ai/
│   │   │   ├── function/        # Function Calling实现
│   │   │   └── rag/             # RAG相关
│   │   ├── config/              # 配置类
│   │   ├── service/             # 业务服务
│   │   └── controller/          # REST API
│   └── src/main/resources/
│       ├── application.yaml.example
│       └── markdown/            # RAG知识库
├── ticket-mcp-server/            # MCP工具服务
│   ├── ticket-mcp-log-service/   # 日志查询
│   └── ticket-mcp-metrics-service/ # 指标监控
├── vue/                         # 前端
├── CONFIG.md                    # 配置文档
├── DEPLOY_TO_GITHUB.md         # 部署指南
└── README.md
```

## 部署说明

详细部署流程请查看 [DEPLOY_TO_GITHUB.md](DEPLOY_TO_GITHUB.md)

## 开发计划

- [ ] 支持更多AI模型 (Claude, Gemini等)
- [ ] 接入向量数据库 (Milvus, Qdrant)
- [ ] 增强Rerank算法
- [ ] 支持流式输出
- [ ] 多租户支持
- [ ] 更丰富的MCP工具

## 技术交流

如有问题或建议，欢迎提Issue讨论。

## License

MIT License
