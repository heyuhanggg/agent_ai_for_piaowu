package org.javaup.ai.config;


import org.javaup.ai.advisor.*;
import org.javaup.ai.ai.rag.OpsKnowledgeLoader;
import org.javaup.ai.ai.function.AiProgram;
import org.javaup.ai.constants.DaMaiConstant;
import org.javaup.ai.enums.ChatType;
import org.javaup.ai.service.AiObservabilityService;
import org.javaup.ai.service.ChatTypeHistoryService;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.ChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.deepseek.DeepSeekChatModel;
import org.springframework.ai.openai.OpenAiEmbeddingModel;
import org.springframework.ai.document.Document;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.util.List;

import static org.javaup.ai.constants.DaMaiConstant.CHAT_TITLE_ADVISOR_ORDER;
import static org.javaup.ai.constants.DaMaiConstant.CHAT_TYPE_HISTORY_ADVISOR_ORDER;
import static org.javaup.ai.constants.DaMaiConstant.MESSAGE_CHAT_MEMORY_ADVISOR_ORDER;
import static org.javaup.ai.constants.DaMaiConstant.OBSERVABILITY_ADVISOR_ORDER;


public class DaMaiAiAutoConfiguration {

    // 最基础的通用 ChatClient，主要用于简单聊天或作为兜底能力。
    @Bean
    public ChatClient chatClient(DeepSeekChatModel model) {
        return ChatClient
                .builder(model)
                // 给通用聊天模型设置一个基础系统角色。
                .defaultSystem("你是一位智能助手，你的特点是温柔、善良，你的名字叫智能小艾，要结合你的特点积极的回答用户的问题。")
                .defaultAdvisors(
                        // 打印请求日志，便于开发调试。
                        new SimpleLoggerAdvisor()
                )
                .build();
    }

    // 统一的会话记忆组件。
    // 三个助手都会使用它，通过 conversationId 区分不同会话，并只保留最近 20 条消息。
    @Bean
    public ChatMemory chatMemory(ChatMemoryRepository chatMemoryRepository) {
        return MessageWindowChatMemory.builder()
                .chatMemoryRepository(chatMemoryRepository)
                .maxMessages(20)
                .build();
    }

    // 贴心助手专用 ChatClient。
    // 采用ReAct决策 + 上下文工程 + 护栏体系 + 智能路由的完整Agent架构
    @Bean
    public ChatClient assistantChatClient(DeepSeekChatModel model, ChatMemory chatMemory, AiProgram aiProgram,
                                          ChatTypeHistoryService chatTypeHistoryService,
                                          @Qualifier("titleChatClient")ChatClient titleChatClient,
                                          AiObservabilityService observabilityService,
                                          VectorStore vectorStore) {
        return ChatClient
                .builder(model)
                .defaultSystem(DaMaiConstant.DA_MAI_SYSTEM_PROMPT)
                .defaultAdvisors(
                        new SimpleLoggerAdvisor(),
                        // === 熔断保护层 === (V3新增)
                        // 熔断器: LLM连续失败5次触发熔断，60秒冷却后半开探测
                        CircuitBreakerAdvisor.builder()
                                .failureThreshold(5)
                                .cooldownSeconds(60)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 55)
                                .build(),
                        // === 护栏层 ===
                        // 输入护栏: Prompt注入检测、敏感信息过滤、输入长度限制
                        InputGuardrailAdvisor.builder()
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 50)
                                .build(),
                        // === 路由层 ===
                        // 智能路由: 自动识别意图(购票/规则/闲聊)并注入增强指令
                        AgentRouterAdvisor.builder()
                                .routerModel(model)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 40)
                                .build(),
                        // === 提示词工程层 === (V3新增)
                        // 结构化提示词: 动态Few-Shot示例注入 + CoT模板 + 角色边界提醒
                        StructuredPromptAdvisor.builder()
                                .agentRole("assistant")
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 38)
                                .build(),
                        // === 上下文工程层 ===
                        // 消息重要度评分: 按信息密度/决策关键性/时间衰减加权保留 (V3新增)
                        ConversationImportanceAdvisor.builder()
                                .maxMessages(20)
                                .protectedRecentCount(4)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 32)
                                .build(),
                        // 上下文压缩: 对话超过12轮时自动摘要早期消息
                        ContextCompressionAdvisor.builder(model)
                                .compressionThreshold(12)
                                .preserveRecentCount(4)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 30)
                                .build(),
                        // 实体记忆: 自动提取用户信息(城市/节目/手机号等)，避免重复询问
                        EntityMemoryAdvisor.builder(model)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 25)
                                .build(),
                        // 工具结果摘要: 压缩超长的工具返回数据
                        ToolResultSummaryAdvisor.builder(model)
                                .toolResultMaxLength(1500)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 20)
                                .build(),
                        // 工具结果缓存: 相同参数的查询复用缓存，写操作不缓存 (V3新增)
                        ToolResultCacheAdvisor.builder()
                                .cacheTtlSeconds(300)
                                .maxCacheSize(15)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 18)
                                .build(),
                        // === 自适应RAG层 ===
                        // 自适应RAG: 规则查询自动触发知识库检索，闲聊/操作不检索
                        AdaptiveRagAdvisor.builder(vectorStore)
                                .routerModel(model)
                                .defaultTopK(3)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 15)
                                .build(),
                        // === 决策层 ===
                        // ReAct模式: Thought→Action→Observation迭代，最多5轮，含重复检测/强制终止/Token预算感知
                        ReactAdvisor.builder()
                                .maxIterations(5)
                                .enableReactLoop(true)
                                .tokenBudgetWarningThreshold(6000)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 10)
                                .build(),
                        // === 业务层 ===
                        ChatTypeHistoryAdvisor.builder(chatTypeHistoryService).type(ChatType.ASSISTANT.getCode())
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER).build(),
                        ChatTypeTitleAdvisor.builder(chatTypeHistoryService).type(ChatType.ASSISTANT.getCode())
                                .chatClient(titleChatClient).chatMemory(chatMemory).order(CHAT_TITLE_ADVISOR_ORDER).build(),
                        MessageChatMemoryAdvisor.builder(chatMemory).order(MESSAGE_CHAT_MEMORY_ADVISOR_ORDER).build(),
                        // === 工具护栏层 ===
                        // 工具调用安全: 频率限制、幂等保护(防重复下单)、调用审计
                        ToolCallGuardrailAdvisor.builder()
                                .maxCallsPerTool(10)
                                .maxTotalCalls(30)
                                .order(OBSERVABILITY_ADVISOR_ORDER - 10)
                                .build(),
                        // === 可观测性层 ===
                        AiObservabilityAdvisor.builder(observabilityService)
                                .order(OBSERVABILITY_ADVISOR_ORDER)
                                .modelName("deepseek-chat")
                                .requestType(ChatType.ASSISTANT.getMsg())
                                .build(),
                        // === 输出护栏层 ===
                        // 输出护栏: 敏感信息脱敏、编造数据检测
                        OutputGuardrailAdvisor.builder()
                                .order(OBSERVABILITY_ADVISOR_ORDER + 10)
                                .build(),
                        // 结构化输出校验: 格式完整性检查 + JSON修复 + 降级提示 (V3新增)
                        StructuredOutputAdvisor.builder()
                                .repairModel(model)
                                .order(OBSERVABILITY_ADVISOR_ORDER + 15)
                                .build(),
                        // 幻觉接地检测: 对比工具返回事实库，标注无来源数据 (V3新增)
                        HallucinationGroundingAdvisor.builder()
                                .enableWarningAnnotation(true)
                                .order(OBSERVABILITY_ADVISOR_ORDER + 20)
                                .build()
                )
                .defaultTools(aiProgram)
                .build();
    }
    
    // 运维知识库文档加载器
    @Bean
    public OpsKnowledgeLoader opsKnowledgeLoader(ResourcePatternResolver resourcePatternResolver) {
        return new OpsKnowledgeLoader(resourcePatternResolver);
    }
    
    // 运维专用向量库（与规则助手的向量库隔离，避免知识混淆）
    @Bean("opsVectorStore")
    public VectorStore opsVectorStore(OpenAiEmbeddingModel embeddingModel, OpsKnowledgeLoader opsKnowledgeLoader) {
        SimpleVectorStore opsStore = SimpleVectorStore.builder(embeddingModel).build();
        // 启动时加载运维知识库文档到向量库
        List<Document> opsDocs = opsKnowledgeLoader.loadOpsKnowledge();
        if (!opsDocs.isEmpty()) {
            opsStore.add(opsDocs);
        }
        return opsStore;
    }
    
    // 运维助手专用 ChatClient。
    // 采用Plan-Execute-Replan模式 + 运维RAG知识库 + 上下文工程 + 护栏体系的完整运维Agent架构
    @Bean
    public ChatClient analysisChatClient(DeepSeekChatModel model, ChatMemory chatMemory,
                                          ChatTypeHistoryService chatTypeHistoryService,
                                          @Qualifier("titleChatClient")ChatClient titleChatClient,
                                          @Qualifier("mcpToolCallbackProvider") ToolCallbackProvider mcpToolCallbackProvider,
                                          @Qualifier("opsVectorStore") VectorStore opsVectorStore,
                                          AiObservabilityService observabilityService) {
        return ChatClient
                .builder(model)
                .defaultSystem(DaMaiConstant.DA_MAI_ANALYSIS_PROMPT)
                .defaultAdvisors(
                        new SimpleLoggerAdvisor(),
                        // === 熔断保护层 === (V3新增)
                        CircuitBreakerAdvisor.builder()
                                .failureThreshold(5)
                                .cooldownSeconds(60)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 55)
                                .build(),
                        // === 护栏层 ===
                        InputGuardrailAdvisor.builder()
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 50)
                                .build(),
                        // === 提示词工程层 === (V3新增)
                        // 结构化提示词: 运维诊断CoT模板 + 日志/链路Few-Shot + 角色边界
                        StructuredPromptAdvisor.builder()
                                .agentRole("analysis")
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 38)
                                .build(),
                        // === 上下文工程层 ===
                        // 消息重要度评分: 工具结果+4分，诊断结论+2分，闲聊-3分 (V3新增)
                        ConversationImportanceAdvisor.builder()
                                .maxMessages(20)
                                .protectedRecentCount(3)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 32)
                                .build(),
                        ContextCompressionAdvisor.builder(model)
                                .compressionThreshold(10)
                                .preserveRecentCount(3)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 30)
                                .build(),
                        EntityMemoryAdvisor.builder(model)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 25)
                                .build(),
                        ToolResultSummaryAdvisor.builder(model)
                                .toolResultMaxLength(2000)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 20)
                                .build(),
                        // 工具结果缓存: MCP查询结果复用，避免重复调用 (V3新增)
                        ToolResultCacheAdvisor.builder()
                                .cacheTtlSeconds(180)
                                .maxCacheSize(20)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 18)
                                .build(),
                        // === 运维RAG知识库层 ===
                        // 运维知识检索: 故障排查手册 + SOP流程 + 历史案例 + 架构文档
                        OpsRagAdvisor.builder(opsVectorStore)
                                .topK(4)
                                .similarityThreshold(0.25)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 15)
                                .build(),
                        // === 错误恢复层 ===
                        // 工具失败自恢复: 超时/服务不可用/无结果/参数错误/权限不足 各类错误按策略恢复
                        // Token预算感知: 边际收益递减时注入收尾指令（借鉴cc-haha-main tokenBudget.ts设计）
                        ErrorRecoveryAdvisor.builder()
                                .maxRecoveryPerType(2)
                                .tokenBudgetWarningThreshold(8000)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 12)
                                .build(),
                        // === 决策层 ===
                        // Plan-Execute-Replan: 制定计划、执行、失败重试、动态调整，最多重规划3次
                        PlanExecuteReplanAdvisor.builder()
                                .maxReplans(3)
                                .enablePlanning(true)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 10)
                                .build(),
                        // === 业务层 ===
                        ChatTypeHistoryAdvisor.builder(chatTypeHistoryService).type(ChatType.ANALYSIS.getCode())
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER).build(),
                        ChatTypeTitleAdvisor.builder(chatTypeHistoryService).type(ChatType.ANALYSIS.getCode())
                                .chatClient(titleChatClient).chatMemory(chatMemory).order(CHAT_TITLE_ADVISOR_ORDER).build(),
                        MessageChatMemoryAdvisor.builder(chatMemory).order(MESSAGE_CHAT_MEMORY_ADVISOR_ORDER).build(),
                        // === 工具护栏层 ===
                        ToolCallGuardrailAdvisor.builder()
                                .maxCallsPerTool(15)
                                .maxTotalCalls(50)
                                .order(OBSERVABILITY_ADVISOR_ORDER - 10)
                                .build(),
                        // === 可观测性层 ===
                        AiObservabilityAdvisor.builder(observabilityService)
                                .order(OBSERVABILITY_ADVISOR_ORDER)
                                .modelName("deepseek-chat")
                                .requestType(ChatType.ANALYSIS.getMsg())
                                .build(),
                        // === 输出护栏层 ===
                        OutputGuardrailAdvisor.builder()
                                .order(OBSERVABILITY_ADVISOR_ORDER + 10)
                                .build(),
                        // 结构化输出校验: 诊断报告格式检查 + JSON修复 (V3新增)
                        StructuredOutputAdvisor.builder()
                                .repairModel(model)
                                .order(OBSERVABILITY_ADVISOR_ORDER + 15)
                                .build(),
                        // 幻觉接地检测: 对比工具查询事实，标注无来源数据 (V3新增)
                        HallucinationGroundingAdvisor.builder()
                                .enableWarningAnnotation(true)
                                .order(OBSERVABILITY_ADVISOR_ORDER + 18)
                                .build(),
                        // === 质量保障层 ===
                        // 自我反思: 完整性/准确性四维自评，对低质量诊断结论注入改进指令
                        SelfReflectionAdvisor.builder(model)
                                .order(OBSERVABILITY_ADVISOR_ORDER + 20)
                                .build()
                )
                .defaultToolCallbacks(mcpToolCallbackProvider)
                .build();
    }
    
    // 标题生成专用的 ChatClient。
    // 这个客户端不直接面向用户，而是给 ChatTypeTitleAdvisor 用于生成会话标题。
    @Bean
    public ChatClient titleChatClient(DeepSeekChatModel model) {
        return ChatClient
                .builder(model)
                .defaultAdvisors(
                        new SimpleLoggerAdvisor()
                )
                .build();
    }
    
    // 向量库组件。
    // 主要给规则助手的 RAG 能力使用，和运维助手不是同一条主链路，但在同一配置类里统一注册。
    @Bean
    public VectorStore vectorStore(OpenAiEmbeddingModel embeddingModel) {
        return SimpleVectorStore.builder(embeddingModel).build();
    }
}
