package org.javaup.ai.config;


import org.javaup.ai.advisor.ChatTypeHistoryAdvisor;
import org.javaup.ai.advisor.ChatTypeTitleAdvisor;
import org.javaup.ai.advisor.ReactAdvisor;
import org.javaup.ai.advisor.PlanExecuteReplanAdvisor;
import org.javaup.ai.ai.function.AiProgram;
import org.javaup.ai.constants.DaMaiConstant;
import org.javaup.ai.enums.ChatType;
import org.javaup.ai.advisor.AiObservabilityAdvisor;
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
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;

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
    // 采用ReAct (Reason + Act) 决策模式，在推理、工具调用与结果观察之间迭代执行
    @Bean
    public ChatClient assistantChatClient(DeepSeekChatModel model, ChatMemory chatMemory, AiProgram aiProgram,
                                          ChatTypeHistoryService chatTypeHistoryService,
                                          @Qualifier("titleChatClient")ChatClient titleChatClient,
                                          AiObservabilityService observabilityService) {
        return ChatClient
                .builder(model)
                .defaultSystem(DaMaiConstant.DA_MAI_SYSTEM_PROMPT)
                .defaultAdvisors(
                        new SimpleLoggerAdvisor(),
                        // ReAct模式: 让模型在思考、行动、观察之间迭代，最多5轮
                        ReactAdvisor.builder()
                                .maxIterations(5)
                                .enableReactLoop(true)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 10)
                                .build(),
                        ChatTypeHistoryAdvisor.builder(chatTypeHistoryService).type(ChatType.ASSISTANT.getCode())
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER).build(),
                        ChatTypeTitleAdvisor.builder(chatTypeHistoryService).type(ChatType.ASSISTANT.getCode())
                                .chatClient(titleChatClient).chatMemory(chatMemory).order(CHAT_TITLE_ADVISOR_ORDER).build(),
                        MessageChatMemoryAdvisor.builder(chatMemory).order(MESSAGE_CHAT_MEMORY_ADVISOR_ORDER).build(),
                        // AI增强: Observability可观测性 - Token统计、延迟监控
                        AiObservabilityAdvisor.builder(observabilityService)
                                .order(OBSERVABILITY_ADVISOR_ORDER)
                                .modelName("deepseek-chat")
                                .requestType(ChatType.ASSISTANT.getMsg())
                                .build()
                )
                .defaultTools(aiProgram)
                .build();
    }
    
    // 运维助手专用 ChatClient。
    // 采用Plan-Execute-Replan模式，制定诊断计划 → 执行步骤 → 观察结果 → 动态调整计划
    @Bean
    public ChatClient analysisChatClient(DeepSeekChatModel model, ChatMemory chatMemory,
                                          ChatTypeHistoryService chatTypeHistoryService,
                                          @Qualifier("titleChatClient")ChatClient titleChatClient,
                                          @Qualifier("mcpToolCallbackProvider") ToolCallbackProvider mcpToolCallbackProvider,
                                          AiObservabilityService observabilityService) {
        return ChatClient
                .builder(model)
                // 给运维助手设置专门的系统提示词，要求它围绕日志查询、链路追踪、系统分析来回答。
                .defaultSystem(DaMaiConstant.DA_MAI_ANALYSIS_PROMPT)
                .defaultAdvisors(
                        // 输出基础交互日志。
                        new SimpleLoggerAdvisor(),
                        // Plan-Execute-Replan模式: 制定计划、执行、根据结果动态调整，最多重规划3次
                        PlanExecuteReplanAdvisor.builder()
                                .maxReplans(3)
                                .enablePlanning(true)
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER - 10)
                                .build(),
                        // 记录当前会话类型为“运维助手”，便于历史会话分类展示。
                        ChatTypeHistoryAdvisor.builder(chatTypeHistoryService).type(ChatType.ANALYSIS.getCode())
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER).build(),
                        // 自动生成当前运维会话的标题，例如“order-service 错误日志排查”。
                        ChatTypeTitleAdvisor.builder(chatTypeHistoryService).type(ChatType.ANALYSIS.getCode())
                                .chatClient(titleChatClient).chatMemory(chatMemory).order(CHAT_TITLE_ADVISOR_ORDER).build(),
                        // 注入多轮会话记忆，让用户继续追问“再看一下 traceId”之类的问题时仍保留上下文。
                        MessageChatMemoryAdvisor.builder(chatMemory).order(MESSAGE_CHAT_MEMORY_ADVISOR_ORDER).build(),
                        // AI 调用可观测性：统计 token、时延、费用、请求类型等数据。
                        AiObservabilityAdvisor.builder(observabilityService)
                                .order(OBSERVABILITY_ADVISOR_ORDER)
                                .modelName("deepseek-chat")
                                .requestType(ChatType.ANALYSIS.getMsg())
                                .build()
                )
                // 挂载 MCP 工具回调。
                // 这样模型在推理过程中就可以真正调用外部工具，而不是只输出“请去查日志”的自然语言建议。
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
