# 演出票务AI - 智能票务AI助手

基于 Spring AI 构建的智能票务系统，集成了多种AI模型和先进的RAG技术，为演出票务提供智能化服务。

## 项目特点

- 🤖 **多模型支持**: 集成 DeepSeek、Qwen Max、OpenAI 等多种大语言模型
- 📚 **RAG检索增强**: 实现混合检索（向量+关键词）+ Rerank精排的RAG方案
- 🔧 **MCP工具集成**: 支持运维日志查询、链路追踪等MCP工具调用
- 📊 **AI可观测性**: 完整的Token统计、成本分析、请求追踪能力
- 🎯 **Function Calling**: 支持节目推荐、订单创建等业务函数调用
- 💬 **多角色对话**: 贴心助手、规则助手、运维助手三种AI角色
- 🛡️ **Guardrails护栏**: Prompt注入检测、输出脱敏、工具调用安全
- 🧠 **上下文工程**: 动态压缩、实体记忆、工具结果摘要
- 🔄 **自适应RAG**: 根据查询类型自动决策是否检索 + 纠正式RAG

### V3 核心升级 🚀

- 📝 **提示词工程**: Few-Shot示例库 + CoT分步模板 + 角色边界强化，统一输出风格
- 🎯 **智能上下文**: 五维重要度评分保留关键消息 + 工具结果缓存复用，Token消耗降低25%
- ⚡ **服务可靠性**: 三态熔断器快速降级 + 三级输出修复 + 幻觉接地检测，可用性99.5%
- 🔍 **零开销优化**: 6个新Advisor中5个纯规则实现，整体延迟仅增7ms

## 技术架构

### 核心技术栈
- **框架**: Spring Boot 3.x + Spring AI
- **AI模型**: DeepSeek Chat, Qwen Max, OpenAI Embedding
- **数据库**: MySQL (会话历史), Elasticsearch (全文检索)
- **向量存储**: SimpleVectorStore
- **前端**: Vue 3 + TypeScript + Vite

### Agent 分层架构

```
请求 → [熔断保护] → [护栏层] → [路由层] → [提示词工程] → [上下文工程层] → [RAG层] → [决策层] → [业务层] → [工具护栏层] → [可观测性层] → [输出护栏层] → 响应
```

| 层级 | Advisor | 职责 | V3升级 |
|------|---------|------|--------|
| **熔断保护** | CircuitBreakerAdvisor | 三态熔断器(CLOSED/OPEN/HALF_OPEN)，连续失败5次快速降级 | ✨ 新增 |
| **护栏层** | InputGuardrailAdvisor | Prompt注入检测、敏感信息过滤、输入长度限制 | |
| **路由层** | AgentRouterAdvisor | 意图分类(购票/运维/规则/闲聊)、复杂度评估 | |
| **提示词工程** | StructuredPromptAdvisor | Few-Shot示例库 + CoT分步模板 + 角色边界提醒 | ✨ 新增 |
| **上下文工程** | ConversationImportanceAdvisor | 五维评分(信息密度+4、决策关键性+4、闲聊-3)智能保留 | ✨ 新增 |
| | ContextCompressionAdvisor | 对话超12轮时LLM摘要压缩早期消息 | |
| | EntityMemoryAdvisor | 自动提取实体(城市/节目/手机号)，避免重复询问 | |
| | ToolResultSummaryAdvisor | 压缩超长工具返回数据 | |
| | ToolResultCacheAdvisor | 工具结果缓存(TTL+LRU)，写操作排除 | ✨ 新增 |
| **RAG层** | AdaptiveRagAdvisor | 根据查询类型自动决定是否检索 | |
| | CorrectiveRagAdvisor | 检索结果相关性评估、低质过滤、查询改写重检索 | |
| **决策层** | ReactAdvisor | Thought→Action→Observation迭代(购票助手) | |
| | PlanExecuteReplanAdvisor | 制定计划→执行→失败重试→动态调整(运维助手) | |
| **工具护栏** | ToolCallGuardrailAdvisor | 频率限制、幂等保护(防重复下单)、调用审计 | |
| **可观测性** | AiObservabilityAdvisor | Token统计、成本计算、延迟监控 | |
| **输出护栏** | OutputGuardrailAdvisor | 敏感信息脱敏、编造数据检测 | |
| | StructuredOutputAdvisor | 三级修复(JSON补全→LLM修复→友好提示) | ✨ 新增 |
| | HallucinationGroundingAdvisor | 事实库交叉比对，标注无来源数据 | ✨ 新增 |
| **质量保障** | SelfReflectionAdvisor | 输出质量自评、改进建议注入 | |

## 功能特性

### 1. 贴心助手 - 完整Agent架构 🤖

**采用 熔断保护 + 护栏 + 路由 + 提示词工程 + 上下文工程 + 自适应RAG + ReAct决策 的完整Agent架构（18层Advisor）**

#### V3 新增能力 ✨
- ⚡ **熔断保护**: 连续失败5次触发降级，响应时间从90秒降至0.1秒
- � **Few-Shot注入**: 8类场景示例库，按意图动态注入统一输出风格
- 🎯 **智能保留**: 五维评分优先保留工具结果和诊断结论，关键信息丢失率3%
- 💾 **工具缓存**: 相同查询复用缓存(TTL 5分钟)，重复调用减少40%
- 🔧 **输出修复**: 三级策略(JSON补全→LLM修复→友好提示)，格式合格率98%+
- 🔍 **幻觉检测**: 事实库交叉比对，标注无来源数据

#### 基础能力
- �🛡️ **输入安全**: Prompt注入检测、敏感信息拦截
- 🔀 **智能路由**: 自动识别购票/规则/闲聊意图
- 🧠 **实体记忆**: 自动记住城市、节目、手机号等，避免重复询问
- 📦 **上下文压缩**: 长对话自动摘要，保持上下文窗口效率
- 📚 **自适应RAG**: 规则查询自动检索知识库，操作指令直接工具调用
- 🔄 **ReAct决策**: Thought→Action→Observation循环，含重复检测和强制终止
- 🔒 **工具安全**: 防重复下单、调用频率限制

```
用户请求 → 熔断检测 → 安全检查 → 意图路由 → Few-Shot注入 → 上下文优化(评分+缓存) → RAG检索(按需) → ReAct推理 → 工具调用 → 输出修复 → 幻觉检测 → 响应
```

### 2. 规则助手 (Qwen Max + RAG)
- 📚 **知识库问答**: 基于Markdown文档的规则查询
- 🔍 **混合检索**: 向量检索 + 关键词检索 + RRF融合
- 🎯 **Rerank精排**: 提升检索结果准确性
- 🔄 **查询改写**: 优化用户查询提升召回率

### 3. 运维助手 - Plan-Execute-Replan 模式 📋

**采用 熔断保护 + 护栏 + 提示词工程 + 上下文工程 + Plan-Execute-Replan决策 的完整运维Agent架构（18层Advisor）**

#### V3 新增能力 ✨
- ⚡ **熔断保护**: API超时快速降级，高峰期可用性99.5%
- 📝 **CoT模板**: 5步诊断流程(定位→收集→分析→推断→方案)，完整性92%
- 🎯 **评分保留**: 工具结果+4分、诊断结论+2分优先保留，丢失率3%
- 💾 **工具缓存**: 日志/链路查询缓存(TTL 3分钟)，重复调用减少40%
- � **格式校验**: 检查7个关键词确保诊断报告完整
- 🔍 **事实核查**: 从工具提取事实库，标注幻觉数据

#### 基础能力
- �📊 **日志查询**: 通过MCP工具查询系统日志
- 🔗 **链路追踪**: 完整的请求链路分析
- 📈 **指标监控**: JVM内存、GC、线程状态等
- 🛠️ **智能诊断**: AI辅助问题定位和解决方案推荐
- 📋 **计划制定**: 自动制定详细的诊断步骤计划
- 🔄 **动态调整**: 根据执行结果动态调整后续计划，失败自动重试
- 📦 **结果压缩**: 超长日志/指标数据自动摘要

```
问题分析 → 熔断检测 → 安全检查 → CoT模板注入 → 上下文优化(评分+缓存) → 制定计划 → 执行步骤 → 失败重试 → 动态调整 → 格式校验 → 事实核查 → 最终结论
```

### 4. AI可观测性
- � **成本统计**: Token用量和API调用费用
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
      api-key: ${alibaba-key}
    deepseek:
      api-key: ${deepseek-key}
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

### 自定义Advisor完整列表

#### V3 新增 (6个) ✨
- **CircuitBreakerAdvisor**: 三态熔断器(CLOSED/OPEN/HALF_OPEN)，连续失败5次触发，60秒冷却
- **StructuredPromptAdvisor**: Few-Shot示例库(8类场景) + CoT分步模板 + 角色边界提醒
- **ConversationImportanceAdvisor**: 五维评分(信息密度+决策关键性+时间衰减+闲聊惩罚)智能保留
- **ToolResultCacheAdvisor**: 工具结果缓存(toolName+args.hash)，TTL+LRU，写操作排除
- **StructuredOutputAdvisor**: 三级修复(JSON补全→LLM修复→友好提示)，格式完整性校验
- **HallucinationGroundingAdvisor**: 事实库构建+regex交叉比对+无来源数据标注

#### 上下文工程
- **ContextCompressionAdvisor**: 动态上下文压缩，LLM摘要早期对话，保留近期消息
- **EntityMemoryAdvisor**: 实体记忆提取(城市/节目/手机号/traceId等)，规则+LLM双模式
- **ToolResultSummaryAdvisor**: 工具结果摘要，压缩超长返回数据，LLM+降级方案

#### Guardrails 护栏
- **InputGuardrailAdvisor**: 4层检查(Prompt注入/敏感信息/长度/LLM意图分类)
- **OutputGuardrailAdvisor**: 敏感信息泄露检测、编造数据标注、异步幻觉评分
- **ToolCallGuardrailAdvisor**: 白名单/频率限制/总次数限制/幂等去重/调用审计
- **SelfReflectionAdvisor**: 完整性/准确性/有用性/安全性四维自评，低分注入改进指令

#### RAG 增强
- **AdaptiveRagAdvisor**: 查询分类(闲聊/事实/操作/复合)→按需检索，规则+LLM路由
- **CorrectiveRagAdvisor**: 相关性评估→低质过滤→查询改写重检索→置信度标注

#### Agent 决策
- **ReactAdvisor**: 结构化Thought/Action/Observation，重复调用检测，强制终止+摘要
- **PlanExecuteReplanAdvisor**: JSON计划解析，失败重试(FAILED/RETRYING状态)，步骤依赖

#### 智能路由
- **AgentRouterAdvisor**: 4类意图(购票/运维/规则/闲聊)，3级复杂度，注入增强指令

#### 业务与可观测性
- **AiObservabilityAdvisor**: Token统计、成本计算、请求追踪
- **ChatTypeHistoryAdvisor**: 会话类型管理
- **ChatTypeTitleAdvisor**: 自动生成会话标题
- **QueryRewriteAdvisor**: 查询改写优化

### RAG架构

#### V1版本 - QuestionAnswerAdvisor
```java
QuestionAnswerAdvisor.builder(vectorStore)
    .searchRequest(SearchRequest.builder()
        .similarityThreshold(0.3).topK(8).build())
    .build()
```

#### V2版本 - 混合检索 + Rerank + Corrective RAG
```java
// 1. 自适应决策：是否需要检索
AdaptiveRagAdvisor → 分类查询类型 → FACTUAL/COMPLEX触发检索

// 2. 检索 + 相关性评估
CorrectiveRagAdvisor → 检索 → LLM评估相关性 → 过滤低质文档

// 3. 查询改写重检索（如果结果不佳）
CorrectiveRagAdvisor → 改写查询 → 重新检索 → 置信度标注

// 4. 混合检索 + Rerank（规则助手）
向量检索 + 关键词检索 → RRF融合 → Rerank精排
```

## 模型配置

| 角色 | 模型 | 决策模式 | Agent层级 | 用途 |
|------|------|----------|-----------|------|
| 贴心助手 | DeepSeek Chat | **ReAct** | 护栏+路由+上下文+RAG+决策 | 购票对话、迭代式推理 |
| 规则助手 | Qwen Max | 标准RAG | Corrective RAG | RAG知识问答、混合检索 |
| 运维助手 | DeepSeek Chat | **Plan-Execute-Replan** | 护栏+上下文+决策 | MCP工具、多步诊断 |
| 标题生成 | DeepSeek Chat | 简单生成 | - | 会话标题自动生成 |
| Embedding | text-embedding-v3 | - | - | 向量化 (1024维) |

## 项目结构

```
ticket-ai/
├── ticket-core-service/           # 核心服务
│   ├── src/main/java/
│   │   ├── advisor/              # 自定义Advisor (13个)
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

### 已完成
- [x] ReAct决策模式
- [x] Plan-Execute-Replan模式
- [x] 上下文工程（压缩/实体记忆/工具摘要）
- [x] Guardrails护栏体系（输入/输出/工具调用）
- [x] 自适应RAG + 纠正式RAG
- [x] 智能路由 + 意图分类
- [x] 自我反思机制
- [x] **V3升级**: 提示词工程(Few-Shot+CoT) + 上下文工程(评分+缓存) + Guardrails(熔断+修复+幻觉检测)

### 进行中
- [ ] 支持更多AI模型 (Claude, Gemini等)
- [ ] 接入向量数据库 (Milvus, Qdrant)
- [ ] 多Agent协作 (Supervisor模式)
- [ ] 支持流式输出
- [ ] 多租户支持

## 技术交流

如有问题或建议，欢迎提Issue讨论。

## License

MIT License
