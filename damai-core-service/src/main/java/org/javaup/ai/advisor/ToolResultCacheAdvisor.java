package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.ToolCall;
import org.springframework.core.Ordered;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 工具结果缓存 Advisor
 *
 * 上下文工程优化——对同一会话中相同参数的工具调用复用缓存结果：
 * 1. 精确匹配缓存：toolName + arguments hash 为 key
 * 2. TTL 过期：默认5分钟自动失效
 * 3. 命中提示：注入提示让模型知道可复用
 * 4. 选择性缓存：写操作（下单/删除）永不缓存
 * 5. 容量控制：LRU淘汰最老缓存
 */
@Slf4j
public class ToolResultCacheAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final long cacheTtlMillis;
    private final int maxCacheSize;

    private static final String CTX_TOOL_CACHE = "tool_result_cache";
    private static final String CTX_CACHE_HITS = "tool_cache_hit_count";

    private static final List<String> WRITE_TOOL_KEYWORDS = List.of(
            "create", "delete", "update", "order", "submit", "cancel",
            "购买", "下单", "取消", "删除", "修改"
    );

    private ToolResultCacheAdvisor(int order, long cacheTtlMillis, int maxCacheSize) {
        this.order = order;
        this.cacheTtlMillis = cacheTtlMillis;
        this.maxCacheSize = maxCacheSize;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        Map<String, Object> context = new HashMap<>(request.context());
        context.putIfAbsent(CTX_TOOL_CACHE, new ConcurrentHashMap<String, CacheEntry>());
        context.putIfAbsent(CTX_CACHE_HITS, 0);

        ConcurrentHashMap<String, CacheEntry> cache =
                (ConcurrentHashMap<String, CacheEntry>) context.get(CTX_TOOL_CACHE);
        long now = Instant.now().toEpochMilli();
        cache.entrySet().removeIf(e -> now - e.getValue().timestamp > cacheTtlMillis);

        if (!cache.isEmpty()) {
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            StringBuilder hint = new StringBuilder("【已缓存的工具结果 - 如需最新数据请说"刷新查询"】\n");
            boolean hasHint = false;
            for (Map.Entry<String, CacheEntry> entry : cache.entrySet()) {
                CacheEntry ce = entry.getValue();
                long ageSec = (now - ce.timestamp) / 1000;
                if (ageSec < 120) {
                    String preview = ce.result.length() > 100 ? ce.result.substring(0, 100) + "..." : ce.result;
                    hint.append(String.format("- %s（%d秒前）: %s\n", ce.toolName, ageSec, preview));
                    hasHint = true;
                }
            }
            if (hasHint) {
                messages.add(new SystemMessage(hint.toString()));
                return ChatClientRequest.builder()
                        .prompt(request.prompt().withInstructions(messages))
                        .context(context)
                        .build();
            }
        }

        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }

    @Override
    @SuppressWarnings("unchecked")
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }

        List<ToolCall> toolCalls = chatResponse.getResult().getToolCalls();
        String output = chatResponse.getResult().getOutput().getText();
        if (toolCalls == null || toolCalls.isEmpty() || output == null) {
            return response;
        }

        Map<String, Object> context = new HashMap<>(response.context());
        ConcurrentHashMap<String, CacheEntry> cache =
                (ConcurrentHashMap<String, CacheEntry>) context.computeIfAbsent(
                        CTX_TOOL_CACHE, k -> new ConcurrentHashMap<>());

        for (ToolCall toolCall : toolCalls) {
            String toolName = toolCall.name();
            if (isWriteOperation(toolName)) {
                log.debug("ToolResultCache: 跳过写操作工具 {}", toolName);
                continue;
            }

            String cacheKey = toolName + ":" + toolCall.arguments().hashCode();
            if (cache.containsKey(cacheKey)) {
                int hits = (Integer) context.getOrDefault(CTX_CACHE_HITS, 0);
                context.put(CTX_CACHE_HITS, hits + 1);
                log.info("ToolResultCache: 命中 {} (累计 {} 次)", toolName, hits + 1);
            } else {
                if (cache.size() >= maxCacheSize) {
                    String oldest = cache.entrySet().stream()
                            .min((a, b) -> Long.compare(a.getValue().timestamp, b.getValue().timestamp))
                            .map(Map.Entry::getKey).orElse(null);
                    if (oldest != null) cache.remove(oldest);
                }
                cache.put(cacheKey, new CacheEntry(toolName, output, Instant.now().toEpochMilli()));
                log.debug("ToolResultCache: 缓存 {} (当前 {} 条)", toolName, cache.size());
            }
        }

        context.put(CTX_TOOL_CACHE, cache);
        return ChatClientResponse.builder().chatResponse(chatResponse).context(context).build();
    }

    private boolean isWriteOperation(String toolName) {
        String lower = toolName.toLowerCase();
        return WRITE_TOOL_KEYWORDS.stream().anyMatch(lower::contains);
    }

    @Override
    public int getOrder() { return order; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 70;
        private long cacheTtlMillis = 5 * 60 * 1000L;
        private int maxCacheSize = 20;

        public Builder order(int order) { this.order = order; return this; }
        public Builder cacheTtlSeconds(int seconds) { this.cacheTtlMillis = seconds * 1000L; return this; }
        public Builder maxCacheSize(int size) { this.maxCacheSize = size; return this; }

        public ToolResultCacheAdvisor build() {
            return new ToolResultCacheAdvisor(order, cacheTtlMillis, maxCacheSize);
        }
    }

    private static class CacheEntry {
        final String toolName;
        final String result;
        final long timestamp;
        CacheEntry(String toolName, String result, long timestamp) {
            this.toolName = toolName;
            this.result = result;
            this.timestamp = timestamp;
        }
    }
}
