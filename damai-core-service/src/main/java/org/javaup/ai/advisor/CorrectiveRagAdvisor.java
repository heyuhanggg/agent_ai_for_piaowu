package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
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
 * 纠正式 RAG Advisor (Corrective RAG / CRAG)
 * 
 * 在标准RAG检索之后增加纠正机制：
 * 1. 相关性评估：使用LLM评估检索文档与查询的相关性
 * 2. 低质量过滤：过滤掉相关性低的文档，避免噪声干扰
 * 3. 查询改写重检索：如果所有文档都不相关，自动改写查询并重新检索
 * 4. 知识补充：标注检索结果的置信度，引导模型合理使用
 * 
 * 流程：检索 → 评估 → (不相关 → 改写 → 重检索) → 注入高质量文档
 */
@Slf4j
public class CorrectiveRagAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final ChatModel evaluationModel;
    private final VectorStore vectorStore;
    private final int topK;
    private final double relevanceThreshold;

    private static final String CTX_CRAG_ORIGINAL_DOCS = "crag_original_docs_count";
    private static final String CTX_CRAG_FILTERED_DOCS = "crag_filtered_docs_count";
    private static final String CTX_CRAG_REWRITTEN = "crag_query_rewritten";

    private static final String RELEVANCE_EVAL_PROMPT = """
            评估以下文档与用户查询的相关性。
            
            用户查询：%s
            
            文档内容：%s
            
            相关性评分（1-10，1=完全不相关，10=高度相关）：
            只输出一个数字。
            """;

    private static final String QUERY_REWRITE_PROMPT = """
            以下查询在知识库中没有找到高相关的文档。请改写查询以提高检索效果。
            
            原始查询：%s
            
            改写策略：
            1. 使用更通用或更精确的关键词
            2. 去掉口语化表达
            3. 补充可能的同义词
            
            只输出改写后的查询（一句话）：
            """;

    private CorrectiveRagAdvisor(int order, ChatModel evaluationModel, VectorStore vectorStore,
                                  int topK, double relevanceThreshold) {
        this.order = order;
        this.evaluationModel = evaluationModel;
        this.vectorStore = vectorStore;
        this.topK = topK;
        this.relevanceThreshold = relevanceThreshold;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        String userInput = request.prompt().getUserMessage().getText();
        Map<String, Object> context = new HashMap<>(request.context());

        // 第一轮检索
        List<Document> docs = search(userInput);
        context.put(CTX_CRAG_ORIGINAL_DOCS, docs.size());

        if (docs.isEmpty()) {
            log.info("CRAG: 首轮检索无结果");
            return request;
        }

        // 评估相关性
        List<ScoredDocument> scoredDocs = evaluateRelevance(userInput, docs);

        // 过滤高相关文档
        List<ScoredDocument> relevant = scoredDocs.stream()
                .filter(d -> d.score >= relevanceThreshold)
                .collect(Collectors.toList());

        log.info("CRAG: 检索到 {} 个文档, 评估后 {} 个高相关 (阈值: {})",
                docs.size(), relevant.size(), relevanceThreshold);

        // 如果没有高相关文档，尝试改写查询重检索
        if (relevant.isEmpty() && evaluationModel != null) {
            String rewrittenQuery = rewriteQuery(userInput);
            if (rewrittenQuery != null && !rewrittenQuery.equals(userInput)) {
                log.info("CRAG: 查询改写 '{}' -> '{}'", truncate(userInput, 40), truncate(rewrittenQuery, 40));
                context.put(CTX_CRAG_REWRITTEN, true);

                List<Document> retryDocs = search(rewrittenQuery);
                if (!retryDocs.isEmpty()) {
                    List<ScoredDocument> retryScoredDocs = evaluateRelevance(rewrittenQuery, retryDocs);
                    relevant = retryScoredDocs.stream()
                            .filter(d -> d.score >= relevanceThreshold * 0.8) // 略放宽阈值
                            .collect(Collectors.toList());
                    log.info("CRAG: 改写后重检索得到 {} 个高相关文档", relevant.size());
                }
            }
        }

        context.put(CTX_CRAG_FILTERED_DOCS, relevant.size());

        if (!relevant.isEmpty()) {
            String ragContext = buildAnnotatedContext(relevant);
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.add(new SystemMessage(ragContext));

            return ChatClientRequest.builder()
                    .prompt(request.prompt().withInstructions(messages))
                    .context(context)
                    .build();
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

    private List<Document> search(String query) {
        try {
            List<Document> results = vectorStore.similaritySearch(
                    SearchRequest.builder()
                            .query(query)
                            .topK(topK)
                            .similarityThreshold(0.2)
                            .build()
            );
            return results != null ? results : List.of();
        } catch (Exception e) {
            log.warn("CRAG检索失败: {}", e.getMessage());
            return List.of();
        }
    }

    private List<ScoredDocument> evaluateRelevance(String query, List<Document> docs) {
        List<ScoredDocument> scored = new ArrayList<>();

        for (Document doc : docs) {
            double score = evaluationModel != null
                    ? llmEvaluate(query, doc.getText())
                    : heuristicEvaluate(query, doc.getText());
            scored.add(new ScoredDocument(doc, score));
        }

        // 按分数降序排列
        scored.sort((a, b) -> Double.compare(b.score, a.score));
        return scored;
    }

    private double llmEvaluate(String query, String docContent) {
        try {
            String prompt = String.format(RELEVANCE_EVAL_PROMPT,
                    truncate(query, 200), truncate(docContent, 500));
            var response = evaluationModel.call(new Prompt(prompt));
            String result = response.getResult().getOutput().getText().trim().replaceAll("[^0-9]", "");
            int score = result.isEmpty() ? 5 : Integer.parseInt(result);
            return Math.min(10, Math.max(1, score)) / 10.0;
        } catch (Exception e) {
            log.debug("LLM相关性评估跳过: {}", e.getMessage());
            return heuristicEvaluate(query, docContent);
        }
    }

    private double heuristicEvaluate(String query, String docContent) {
        if (query == null || docContent == null) return 0.0;

        String[] queryWords = query.toLowerCase().split("[\\s，。、！？]+");
        String lowerDoc = docContent.toLowerCase();
        int matchCount = 0;

        for (String word : queryWords) {
            if (word.length() >= 2 && lowerDoc.contains(word)) {
                matchCount++;
            }
        }

        return queryWords.length > 0 ? (double) matchCount / queryWords.length : 0.0;
    }

    private String rewriteQuery(String originalQuery) {
        try {
            String prompt = String.format(QUERY_REWRITE_PROMPT, originalQuery);
            var response = evaluationModel.call(new Prompt(prompt));
            return response.getResult().getOutput().getText().trim();
        } catch (Exception e) {
            log.debug("查询改写失败: {}", e.getMessage());
            return null;
        }
    }

    private String buildAnnotatedContext(List<ScoredDocument> docs) {
        StringBuilder sb = new StringBuilder("【知识库参考(经相关性评估)】\n");
        for (int i = 0; i < docs.size(); i++) {
            ScoredDocument sd = docs.get(i);
            String confidence = sd.score >= 0.8 ? "高" : sd.score >= 0.5 ? "中" : "低";
            sb.append(String.format("\n[文档%d 置信度:%s]\n%s\n", i + 1, confidence, sd.doc.getText()));
        }
        sb.append("\n请优先参考高置信度文档，低置信度文档仅供参考。如果文档信息与工具查询结果矛盾，以工具查询结果为准。");
        return sb.toString();
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

    private static class ScoredDocument {
        final Document doc;
        final double score;

        ScoredDocument(Document doc, double score) {
            this.doc = doc;
            this.score = score;
        }
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 75;
        private ChatModel evaluationModel;
        private final VectorStore vectorStore;
        private int topK = 5;
        private double relevanceThreshold = 0.5;

        private Builder(VectorStore vectorStore) {
            this.vectorStore = vectorStore;
        }

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder evaluationModel(ChatModel evaluationModel) {
            this.evaluationModel = evaluationModel;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder relevanceThreshold(double relevanceThreshold) {
            this.relevanceThreshold = relevanceThreshold;
            return this;
        }

        public CorrectiveRagAdvisor build() {
            return new CorrectiveRagAdvisor(order, evaluationModel, vectorStore,
                    topK, relevanceThreshold);
        }
    }
}
