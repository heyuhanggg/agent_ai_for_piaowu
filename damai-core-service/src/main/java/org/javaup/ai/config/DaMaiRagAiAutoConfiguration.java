package org.javaup.ai.config;

import org.javaup.ai.advisor.ChatTypeHistoryAdvisor;
import org.javaup.ai.advisor.ChatTypeTitleAdvisor;
import org.javaup.ai.advisor.QueryRewriteAdvisor;
import org.javaup.ai.ai.rag.MarkdownLoader;
import org.javaup.ai.enums.ChatType;
import org.javaup.ai.advisor.AiObservabilityAdvisor;
import org.javaup.ai.service.AiObservabilityService;
import org.javaup.ai.service.ChatTypeHistoryService;
import org.javaup.ai.service.HybridSearchService;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.document.Document;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.core.Ordered;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.util.List;

import static org.javaup.ai.constants.DaMaiConstant.CHAT_TITLE_ADVISOR_ORDER;
import static org.javaup.ai.constants.DaMaiConstant.CHAT_TYPE_HISTORY_ADVISOR_ORDER;
import static org.javaup.ai.constants.DaMaiConstant.MARK_DOWN_SYSTEM_PROMPT;
import static org.javaup.ai.constants.DaMaiConstant.MESSAGE_CHAT_MEMORY_ADVISOR_ORDER;
import static org.javaup.ai.constants.DaMaiConstant.OBSERVABILITY_ADVISOR_ORDER;
import static org.javaup.ai.constants.DaMaiConstant.RAG_VERSION;


@AutoConfigureAfter(DaMaiAiAutoConfiguration.class)
public class DaMaiRagAiAutoConfiguration {
    
    // Markdown 文档加载器：负责把规则知识库里的 .md 文件读取并转成可写入向量库的 Document。
    @Bean
    public MarkdownLoader markdownLoader(ResourcePatternResolver resourcePatternResolver){
        return new MarkdownLoader(resourcePatternResolver);
    }
    
    // 规则助手 V1 版 ChatClient。
    // 这一版采用 Spring AI 自带的 QuestionAnswerAdvisor，让向量库负责相似度检索并把上下文注入给模型。
    @Bean("markdownChatClient")
    @ConditionalOnProperty(name = RAG_VERSION, havingValue = "1",matchIfMissing = true)
    public ChatClient markdownChatClientV1(OpenAiChatModel model, ChatMemory chatMemory, VectorStore vectorStore,
                                           MarkdownLoader markdownLoader, ChatTypeHistoryService chatTypeHistoryService,
                                           @Qualifier("titleChatClient")ChatClient titleChatClient,
                                           AiObservabilityService observabilityService) {
        // 启动时先把 markdown 规则文档全部加载出来。
        List<Document> documentList = markdownLoader.loadMarkdowns();
        // 把规则文档写入向量库，供后续 QuestionAnswerAdvisor 检索。
        vectorStore.add(documentList);
        
        // 构建规则助手专用 ChatClient。
        return ChatClient
                .builder(model)
                // 设置规则助手系统提示词，约束模型回答风格和边界。
                .defaultSystem(MARK_DOWN_SYSTEM_PROMPT)
                .defaultAdvisors(
                        // 打印基础请求日志，便于开发调试。
                        new SimpleLoggerAdvisor(),
                        // 记录当前对话属于“规则助手”类型，用于历史会话和分类统计。
                        ChatTypeHistoryAdvisor.builder(chatTypeHistoryService).type(ChatType.MARKDOWN.getCode())
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER).build(),
                        // 生成会话标题，便于在前端历史记录里展示。
                        ChatTypeTitleAdvisor.builder(chatTypeHistoryService).type(ChatType.MARKDOWN.getCode())
                                .chatClient(titleChatClient).chatMemory(chatMemory).order(CHAT_TITLE_ADVISOR_ORDER).build(),
                        // 注入多轮对话记忆，让规则问答保留上下文。
                        MessageChatMemoryAdvisor.builder(chatMemory).order(MESSAGE_CHAT_MEMORY_ADVISOR_ORDER).build(),
                        // V1 的核心 RAG 能力：从向量库里取相似文档，并把检索结果作为问答上下文交给模型。
                        QuestionAnswerAdvisor.builder(vectorStore)
                                .searchRequest(SearchRequest.builder()
                                        // 相似度阈值：过低容易引入噪声，过高又可能召回不足。
                                        .similarityThreshold(0.3)
                                        // 每次最多取前 8 个最相似文档片段。
                                        .topK(8)
                                        .build())
                                .build(),
                        // 可观测性增强：统计 token、费用、调用类型等信息，便于后续做 AI 运营分析。
                        AiObservabilityAdvisor.builder(observabilityService)
                                .order(OBSERVABILITY_ADVISOR_ORDER)
                                .modelName("qwen-max-latest")
                                .requestType(ChatType.MARKDOWN.getMsg())
                                .build()
                )
                .build();
    }

    // 规则助手 V2 版 ChatClient。
    // 这一版不再依赖 QuestionAnswerAdvisor 自动检索，而是把“查询改写 + 混合检索 + 手工拼接上下文”拆到外层流程中实现。
    @Bean("markdownChatClient")
    @ConditionalOnProperty(name = RAG_VERSION, havingValue = "2")
    public ChatClient markdownChatClientV2(OpenAiChatModel model, ChatMemory chatMemory, VectorStore vectorStore,
                                           MarkdownLoader markdownLoader, ChatTypeHistoryService chatTypeHistoryService,
                                           @Qualifier("titleChatClient")ChatClient titleChatClient,
                                           HybridSearchService hybridSearchService,
                                           AiObservabilityService observabilityService) {
        // 同样先加载 markdown 规则文档。
        List<Document> documentList = markdownLoader.loadMarkdowns();
        // 文档依然会写入向量库，便于混合检索时复用向量能力。
        vectorStore.add(documentList);
        
        // 把文档缓存到 HybridSearchService 中，供关键词检索 + 向量检索混合召回。
        hybridSearchService.cacheDocuments(documentList);
        
        return ChatClient
                .builder(model)
                // 规则助手统一系统提示词。
                .defaultSystem(MARK_DOWN_SYSTEM_PROMPT)
                .defaultAdvisors(
                        new SimpleLoggerAdvisor(),
                        // 在真正检索前先做 Query 改写，提升规则类问答的召回效果。
                        QueryRewriteAdvisor.builder()
                                .order(Ordered.HIGHEST_PRECEDENCE + 50)
                                .enableLLMRewrite(false)  
                                .build(),
                        // 标记当前会话类型为规则助手。
                        ChatTypeHistoryAdvisor.builder(chatTypeHistoryService).type(ChatType.MARKDOWN.getCode())
                                .order(CHAT_TYPE_HISTORY_ADVISOR_ORDER).build(),
                        // 自动生成规则助手会话标题。
                        ChatTypeTitleAdvisor.builder(chatTypeHistoryService).type(ChatType.MARKDOWN.getCode())
                                .chatClient(titleChatClient).chatMemory(chatMemory).order(CHAT_TITLE_ADVISOR_ORDER).build(),
                        // 注入多轮上下文记忆。
                        MessageChatMemoryAdvisor.builder(chatMemory).order(MESSAGE_CHAT_MEMORY_ADVISOR_ORDER).build(),
                        // 可观测性统计。
                        AiObservabilityAdvisor.builder(observabilityService)
                                .order(OBSERVABILITY_ADVISOR_ORDER)
                                .modelName("qwen-max-latest")
                                .requestType(ChatType.MARKDOWN.getMsg())
                                .build()
                )
                .build();
    }
}
