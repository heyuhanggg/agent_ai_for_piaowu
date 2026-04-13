package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.Ordered;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 运维知识库 RAG Advisor
 * 
 * 在运维助手的 Plan-Execute-Replan 流程前，自动检索运维知识库中的相关经验知识，
 * 将故障排查手册、SOP流程、历史案例等注入到上下文中，辅助模型进行诊断分析。
 * 
 * 工作流程：
 * 1. 判断用户查询是否需要运维知识库辅助（运维意图检测）
 * 2. 执行混合检索（向量检索 + 关键词匹配）
 * 3. 将检索到的运维知识注入到对话上下文中
 * 4. 模型在 Plan 阶段结合实时数据 + 经验知识进行诊断
 */
@Slf4j
public class OpsRagAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final VectorStore opsVectorStore;
    private final int topK;
    private final double similarityThreshold;
    private final boolean enableKeywordBoost;

    private static final String CTX_OPS_RAG_TRIGGERED = "ops_rag_triggered";
    private static final String CTX_OPS_RAG_DOCS_COUNT = "ops_rag_docs_count";
    private static final String CTX_OPS_RAG_CATEGORIES = "ops_rag_categories";

    // 运维意图关键词集合
    private static final List<String> OPS_INTENT_KEYWORDS = List.of(
            // 故障排查类
            "报错", "异常", "错误", "error", "exception", "故障", "问题",
            "失败", "不可用", "崩溃", "宕机", "挂了",
            // 性能类
            "慢", "超时", "timeout", "延迟", "卡", "响应时间",
            "性能", "瓶颈",
            // 资源类
            "oom", "内存", "cpu", "磁盘", "连接池", "线程",
            "gc", "堆内存", "full gc",
            // 中间件类
            "redis", "数据库", "mysql", "缓存", "消息队列", "mq",
            "连接断开", "连接失败",
            // 运维操作类
            "发布", "回滚", "扩容", "缩容", "重启", "部署",
            // 分析类
            "排查", "诊断", "分析", "定位", "根因", "原因",
            "怎么办", "怎么处理", "如何解决",
            // 架构类
            "架构", "依赖", "拓扑", "调用链", "链路",
            // 监控告警类
            "告警", "监控", "指标", "巡检"
    );

    // 运维领域同义词映射，用于查询扩展
    private static final Map<String, List<String>> QUERY_EXPANSION_MAP = Map.ofEntries(
            Map.entry("oom", List.of("OutOfMemoryError", "内存溢出", "堆内存耗尽")),
            Map.entry("内存溢出", List.of("OOM", "OutOfMemoryError", "heap space")),
            Map.entry("空指针", List.of("NullPointerException", "NPE")),
            Map.entry("npe", List.of("NullPointerException", "空指针异常")),
            Map.entry("慢sql", List.of("慢查询", "SQL超时", "数据库慢")),
            Map.entry("缓存雪崩", List.of("缓存同时过期", "缓存批量失效")),
            Map.entry("缓存穿透", List.of("缓存未命中", "查询不存在的数据")),
            Map.entry("缓存击穿", List.of("热点key过期", "热点缓存失效")),
            Map.entry("连接池", List.of("connection pool", "HikariPool", "数据库连接")),
            Map.entry("超时", List.of("timeout", "响应慢", "请求超时")),
            Map.entry("回滚", List.of("rollback", "版本回退", "发布回滚"))
    );

    private OpsRagAdvisor(int order, VectorStore opsVectorStore, int topK,
                           double similarityThreshold, boolean enableKeywordBoost) {
        this.order = order;
        this.opsVectorStore = opsVectorStore;
        this.topK = topK;
        this.similarityThreshold = similarityThreshold;
        this.enableKeywordBoost = enableKeywordBoost;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        String userInput = request.prompt().getUserMessage().getText();
        Map<String, Object> context = new HashMap<>(request.context());

        // 判断是否需要检索运维知识库
        if (!isOpsRelatedQuery(userInput)) {
            context.put(CTX_OPS_RAG_TRIGGERED, false);
            log.debug("运维RAG：非运维相关查询，跳过知识库检索");
            return ChatClientRequest.builder()
                    .prompt(request.prompt())
                    .context(context)
                    .build();
        }

        // 查询扩展：补充同义词提升召回
        String expandedQuery = expandQuery(userInput);
        log.info("运维RAG：原始查询=[{}]，扩展查询=[{}]", truncate(userInput, 60), truncate(expandedQuery, 100));

        // 执行向量检索
        List<Document> docs = retrieveOpsKnowledge(expandedQuery);

        if (docs.isEmpty()) {
            context.put(CTX_OPS_RAG_TRIGGERED, false);
            log.info("运维RAG：未检索到相关运维知识");
            return ChatClientRequest.builder()
                    .prompt(request.prompt())
                    .context(context)
                    .build();
        }

        // 构建运维知识上下文并注入
        String opsContext = buildOpsContext(docs);
        Set<String> categories = docs.stream()
                .map(d -> (String) d.getMetadata().getOrDefault("category", "general"))
                .collect(Collectors.toSet());

        List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
        messages.add(new SystemMessage(opsContext));

        context.put(CTX_OPS_RAG_TRIGGERED, true);
        context.put(CTX_OPS_RAG_DOCS_COUNT, docs.size());
        context.put(CTX_OPS_RAG_CATEGORIES, categories);

        log.info("运维RAG：检索到 {} 个相关文档，类别: {}", docs.size(), categories);

        return ChatClientRequest.builder()
                .prompt(request.prompt().withInstructions(messages))
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        return response;
    }

    /**
     * 判断用户查询是否与运维相关
     */
    private boolean isOpsRelatedQuery(String input) {
        if (input == null || input.trim().length() < 3) {
            return false;
        }
        String lower = input.toLowerCase();

        // 匹配运维意图关键词（命中任意一个即认为与运维相关）
        int matchCount = 0;
        for (String keyword : OPS_INTENT_KEYWORDS) {
            if (lower.contains(keyword)) {
                matchCount++;
            }
        }

        // 至少命中1个关键词，或者查询长度较长（可能是复杂问题描述）
        return matchCount >= 1 || input.length() > 80;
    }

    /**
     * 查询扩展：基于同义词映射补充关键词
     */
    private String expandQuery(String query) {
        String lower = query.toLowerCase();
        StringBuilder expanded = new StringBuilder(query);

        for (Map.Entry<String, List<String>> entry : QUERY_EXPANSION_MAP.entrySet()) {
            if (lower.contains(entry.getKey())) {
                for (String synonym : entry.getValue()) {
                    if (!lower.contains(synonym.toLowerCase())) {
                        expanded.append(" ").append(synonym);
                    }
                }
            }
        }

        return expanded.toString();
    }

    /**
     * 从运维知识库中检索相关文档
     */
    private List<Document> retrieveOpsKnowledge(String query) {
        try {
            List<Document> results = opsVectorStore.similaritySearch(
                    SearchRequest.builder()
                            .query(query)
                            .topK(topK)
                            .similarityThreshold(similarityThreshold)
                            .build()
            );
            return results != null ? results : List.of();
        } catch (Exception e) {
            log.warn("运维RAG：知识库检索失败: {}", e.getMessage());
            return List.of();
        }
    }

    /**
     * 构建运维知识上下文
     */
    private String buildOpsContext(List<Document> docs) {
        StringBuilder sb = new StringBuilder();
        sb.append("【运维知识库参考】以下是从运维知识库中检索到的相关经验知识，请结合这些信息和工具查询的实时数据进行诊断分析：\n\n");

        for (int i = 0; i < docs.size(); i++) {
            Document doc = docs.get(i);
            String category = (String) doc.getMetadata().getOrDefault("category", "general");
            String source = (String) doc.getMetadata().getOrDefault("name", "未知来源");
            String categoryLabel = getCategoryLabel(category);

            sb.append(String.format("--- 参考文档 %d [%s - %s] ---\n", i + 1, categoryLabel, source));
            sb.append(doc.getText());
            sb.append("\n\n");
        }

        sb.append("---\n");
        sb.append("【使用说明】\n");
        sb.append("- 以上知识仅作参考，具体问题需要结合工具查询的实时数据进行判断\n");
        sb.append("- 引用知识库内容时请标注来源，如\"根据运维手册...\"、\"参考历史案例...\"\n");
        sb.append("- 如果知识库内容与实时数据矛盾，以实时数据为准\n");

        return sb.toString();
    }

    /**
     * 获取文档类别的中文标签
     */
    private String getCategoryLabel(String category) {
        return switch (category) {
            case "troubleshooting" -> "故障排查手册";
            case "sop" -> "SOP操作流程";
            case "case_study" -> "历史故障案例";
            case "architecture" -> "架构说明";
            default -> "运维知识";
        };
    }

    private String truncate(String str, int maxLen) {
        if (str == null) return "";
        return str.length() > maxLen ? str.substring(0, maxLen) + "..." : str;
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder(VectorStore opsVectorStore) {
        return new Builder(opsVectorStore);
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 70;
        private final VectorStore opsVectorStore;
        private int topK = 4;
        private double similarityThreshold = 0.25;
        private boolean enableKeywordBoost = true;

        private Builder(VectorStore opsVectorStore) {
            this.opsVectorStore = opsVectorStore;
        }

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder similarityThreshold(double similarityThreshold) {
            this.similarityThreshold = similarityThreshold;
            return this;
        }

        public Builder enableKeywordBoost(boolean enableKeywordBoost) {
            this.enableKeywordBoost = enableKeywordBoost;
            return this;
        }

        public OpsRagAdvisor build() {
            return new OpsRagAdvisor(order, opsVectorStore, topK,
                    similarityThreshold, enableKeywordBoost);
        }
    }
}
