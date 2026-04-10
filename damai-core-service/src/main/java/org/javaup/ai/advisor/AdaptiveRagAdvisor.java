package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 自适应 RAG Advisor
 * 
 * 根据用户查询类型自动决定是否需要RAG检索，以及检索策略：
 * 1. 路由决策：分类用户查询 → 闲聊/事实查询/规则查询/操作指令
 * 2. 闲聊/操作指令 → 不检索，直接通过
 * 3. 事实查询/规则查询 → 触发RAG检索，注入检索结果
 * 4. 支持动态调整检索参数（topK、相似度阈值）
 * 
 * 这样可以避免在不需要检索时浪费时间和token。
 */
@Slf4j
public class AdaptiveRagAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final ChatModel routerModel;
    private final VectorStore vectorStore;
    private final int defaultTopK;
    private final double similarityThreshold;

    private static final String CTX_RAG_DECISION = "adaptive_rag_decision";
    private static final String CTX_RAG_DOCS_COUNT = "adaptive_rag_docs_count";

    private static final String ROUTER_PROMPT = """
            判断以下用户输入属于哪种类型，只输出类别编号：
            
            1. CHITCHAT - 闲聊问候（你好、谢谢、再见等）
            2. FACTUAL - 事实/规则查询（退票规则、购票须知、优惠政策等）
            3. ACTION - 操作指令（推荐节目、查询票价、下单、查日志等）
            4. COMPLEX - 需要知识库+工具的复合查询
            
            用户输入：%s
            
            只输出：1、2、3 或 4
            """;

    private AdaptiveRagAdvisor(int order, ChatModel routerModel, VectorStore vectorStore,
                                int defaultTopK, double similarityThreshold) {
        this.order = order;
        this.routerModel = routerModel;
        this.vectorStore = vectorStore;
        this.defaultTopK = defaultTopK;
        this.similarityThreshold = similarityThreshold;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        String userInput = request.prompt().getUserMessage().getText();
        Map<String, Object> context = new HashMap<>(request.context());

        // 先用规则快速判断
        QueryType queryType = quickClassify(userInput);

        // 模糊情况使用LLM判断
        if (queryType == QueryType.UNKNOWN && routerModel != null) {
            queryType = llmClassify(userInput);
        }

        context.put(CTX_RAG_DECISION, queryType.name());
        log.info("自适应RAG决策: {} -> {}", truncate(userInput, 50), queryType);

        // 只有FACTUAL和COMPLEX类型触发RAG检索
        if (queryType == QueryType.FACTUAL || queryType == QueryType.COMPLEX) {
            List<Document> docs = retrieveDocuments(userInput, queryType);

            if (!docs.isEmpty()) {
                String ragContext = buildRagContext(docs);
                List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
                messages.add(new SystemMessage(ragContext));
                context.put(CTX_RAG_DOCS_COUNT, docs.size());

                log.info("自适应RAG: 检索到 {} 个相关文档", docs.size());

                return ChatClientRequest.builder()
                        .prompt(request.prompt().withInstructions(messages))
                        .context(context)
                        .build();
            } else {
                log.info("自适应RAG: 未检索到相关文档，降级为无RAG模式");
            }
        }

        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        return response;
    }

    private QueryType quickClassify(String input) {
        if (input == null || input.trim().length() < 3) {
            return QueryType.CHITCHAT;
        }

        String lower = input.toLowerCase();

        // 闲聊模式
        if (lower.matches("^(你好|hi|hello|谢谢|感谢|再见|拜拜|嗯|好的|ok|行|明白了).*")) {
            return QueryType.CHITCHAT;
        }

        // 规则/事实查询
        if (lower.contains("规则") || lower.contains("政策") || lower.contains("须知") ||
            lower.contains("怎么退") || lower.contains("可以退") || lower.contains("退票") ||
            lower.contains("购票须知") || lower.contains("注意事项") || lower.contains("优惠")) {
            return QueryType.FACTUAL;
        }

        // 操作指令
        if (lower.contains("推荐") || lower.contains("查询") || lower.contains("下单") ||
            lower.contains("买票") || lower.contains("查日志") || lower.contains("链路") ||
            lower.contains("监控") || lower.contains("分析")) {
            return QueryType.ACTION;
        }

        return QueryType.UNKNOWN;
    }

    private QueryType llmClassify(String input) {
        try {
            String prompt = String.format(ROUTER_PROMPT, truncate(input, 200));
            var response = routerModel.call(new Prompt(prompt));
            String result = response.getResult().getOutput().getText().trim();

            return switch (result) {
                case "1" -> QueryType.CHITCHAT;
                case "2" -> QueryType.FACTUAL;
                case "3" -> QueryType.ACTION;
                case "4" -> QueryType.COMPLEX;
                default -> QueryType.ACTION;
            };
        } catch (Exception e) {
            log.debug("LLM路由分类跳过: {}", e.getMessage());
            return QueryType.ACTION;
        }
    }

    private List<Document> retrieveDocuments(String query, QueryType type) {
        try {
            int topK = type == QueryType.COMPLEX ? defaultTopK + 2 : defaultTopK;

            List<Document> results = vectorStore.similaritySearch(
                    SearchRequest.builder()
                            .query(query)
                            .topK(topK)
                            .similarityThreshold(similarityThreshold)
                            .build()
            );
            return results != null ? results : List.of();
        } catch (Exception e) {
            log.warn("RAG检索失败: {}", e.getMessage());
            return List.of();
        }
    }

    private String buildRagContext(List<Document> docs) {
        String content = docs.stream()
                .map(Document::getText)
                .collect(Collectors.joining("\n---\n"));

        return "【知识库参考】以下是从知识库中检索到的相关信息，请结合这些信息回答用户问题：\n" + content;
    }

    private String truncate(String str, int maxLen) {
        if (str == null) return "";
        return str.length() > maxLen ? str.substring(0, maxLen) + "..." : str;
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder(VectorStore vectorStore) {
        return new Builder(vectorStore);
    }

    private enum QueryType {
        CHITCHAT,   // 闲聊
        FACTUAL,    // 事实/规则查询
        ACTION,     // 操作指令
        COMPLEX,    // 复合查询
        UNKNOWN     // 待判断
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 70;
        private ChatModel routerModel;
        private final VectorStore vectorStore;
        private int defaultTopK = 3;
        private double similarityThreshold = 0.3;

        private Builder(VectorStore vectorStore) {
            this.vectorStore = vectorStore;
        }

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder routerModel(ChatModel routerModel) {
            this.routerModel = routerModel;
            return this;
        }

        public Builder defaultTopK(int defaultTopK) {
            this.defaultTopK = defaultTopK;
            return this;
        }

        public Builder similarityThreshold(double similarityThreshold) {
            this.similarityThreshold = similarityThreshold;
            return this;
        }

        public AdaptiveRagAdvisor build() {
            return new AdaptiveRagAdvisor(order, routerModel, vectorStore,
                    defaultTopK, similarityThreshold);
        }
    }
}
