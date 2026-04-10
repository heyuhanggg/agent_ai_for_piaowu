package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.ToolCall;
import org.springframework.core.Ordered;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 工具调用护栏 Advisor
 * 
 * 对模型发起的工具调用进行安全性和合理性检查：
 * 1. 权限控制：按角色限制可调用的工具白名单
 * 2. 频率限制：防止工具被频繁调用（如下单接口）
 * 3. 参数验证：检查工具调用参数的合理性
 * 4. 调用审计：记录所有工具调用，便于事后审查
 * 5. 幂等保护：对写操作进行去重，防止重复下单
 */
@Slf4j
public class ToolCallGuardrailAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final Set<String> allowedTools;
    private final Set<String> dangerousTools;
    private final int maxCallsPerTool;
    private final int maxTotalCalls;

    private static final String CTX_TOOL_CALL_LOG = "tool_call_audit_log";
    private static final String CTX_TOOL_CALL_BLOCKED = "tool_call_blocked";

    // 每个会话的工具调用计数
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, AtomicInteger>> callCounters;
    // 幂等去重：记录写操作的参数hash
    private final ConcurrentHashMap<String, Set<String>> writeOperationHashes;

    private ToolCallGuardrailAdvisor(int order, Set<String> allowedTools, 
                                      Set<String> dangerousTools,
                                      int maxCallsPerTool, int maxTotalCalls) {
        this.order = order;
        this.allowedTools = allowedTools;
        this.dangerousTools = dangerousTools;
        this.maxCallsPerTool = maxCallsPerTool;
        this.maxTotalCalls = maxTotalCalls;
        this.callCounters = new ConcurrentHashMap<>();
        this.writeOperationHashes = new ConcurrentHashMap<>();
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        return request;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }

        List<ToolCall> toolCalls = chatResponse.getResult().getToolCalls();
        if (toolCalls == null || toolCalls.isEmpty()) {
            return response;
        }

        Map<String, Object> context = new HashMap<>(response.context());
        String conversationId = getConversationId(context, "default");
        List<Map<String, String>> auditLog = (List<Map<String, String>>) 
                context.getOrDefault(CTX_TOOL_CALL_LOG, new ArrayList<>());

        ConcurrentHashMap<String, AtomicInteger> sessionCounters = 
                callCounters.computeIfAbsent(conversationId, k -> new ConcurrentHashMap<>());
        Set<String> sessionHashes = 
                writeOperationHashes.computeIfAbsent(conversationId, k -> ConcurrentHashMap.newKeySet());

        List<String> blockedCalls = new ArrayList<>();

        for (ToolCall toolCall : toolCalls) {
            String toolName = toolCall.name();
            String toolArgs = toolCall.arguments();

            // 审计记录
            Map<String, String> auditEntry = new HashMap<>();
            auditEntry.put("tool", toolName);
            auditEntry.put("args_hash", String.valueOf(toolArgs.hashCode()));
            auditEntry.put("timestamp", String.valueOf(System.currentTimeMillis()));

            // 检查1：白名单
            if (!allowedTools.isEmpty() && !allowedTools.contains(toolName)) {
                log.warn("工具护栏 - 工具不在白名单中: {}", toolName);
                auditEntry.put("status", "BLOCKED_NOT_ALLOWED");
                blockedCalls.add(toolName + " (未授权)");
                auditLog.add(auditEntry);
                continue;
            }

            // 检查2：单工具频率限制
            int callCount = sessionCounters
                    .computeIfAbsent(toolName, k -> new AtomicInteger(0))
                    .incrementAndGet();
            if (callCount > maxCallsPerTool) {
                log.warn("工具护栏 - 工具 {} 调用次数超限: {}/{}", toolName, callCount, maxCallsPerTool);
                auditEntry.put("status", "BLOCKED_RATE_LIMIT");
                blockedCalls.add(toolName + " (频率超限)");
                auditLog.add(auditEntry);
                continue;
            }

            // 检查3：总调用次数限制
            int totalCalls = sessionCounters.values().stream()
                    .mapToInt(AtomicInteger::get).sum();
            if (totalCalls > maxTotalCalls) {
                log.warn("工具护栏 - 总调用次数超限: {}/{}", totalCalls, maxTotalCalls);
                auditEntry.put("status", "BLOCKED_TOTAL_LIMIT");
                blockedCalls.add(toolName + " (总次数超限)");
                auditLog.add(auditEntry);
                continue;
            }

            // 检查4：危险工具幂等去重
            if (dangerousTools.contains(toolName)) {
                String argsHash = toolName + ":" + toolArgs.hashCode();
                if (sessionHashes.contains(argsHash)) {
                    log.warn("工具护栏 - 检测到重复的写操作: {} (参数相同)", toolName);
                    auditEntry.put("status", "BLOCKED_DUPLICATE");
                    blockedCalls.add(toolName + " (重复操作)");
                    auditLog.add(auditEntry);
                    continue;
                }
                sessionHashes.add(argsHash);
                log.info("工具护栏 - 危险工具调用已记录: {}", toolName);
            }

            auditEntry.put("status", "ALLOWED");
            auditLog.add(auditEntry);
            log.debug("工具护栏 - 允许调用: {} (第{}次)", toolName, callCount);
        }

        context.put(CTX_TOOL_CALL_LOG, auditLog);

        if (!blockedCalls.isEmpty()) {
            context.put(CTX_TOOL_CALL_BLOCKED, blockedCalls);
            log.info("工具护栏 - 本轮拦截 {} 个工具调用: {}", blockedCalls.size(), blockedCalls);
        }

        return ChatClientResponse.builder()
                .chatResponse(chatResponse)
                .context(context)
                .build();
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 80;
        private Set<String> allowedTools = new HashSet<>();
        private Set<String> dangerousTools = Set.of("createOrder", "create_order");
        private int maxCallsPerTool = 10;
        private int maxTotalCalls = 30;

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder allowedTools(Set<String> allowedTools) {
            this.allowedTools = allowedTools;
            return this;
        }

        public Builder dangerousTools(Set<String> dangerousTools) {
            this.dangerousTools = dangerousTools;
            return this;
        }

        public Builder maxCallsPerTool(int maxCallsPerTool) {
            this.maxCallsPerTool = maxCallsPerTool;
            return this;
        }

        public Builder maxTotalCalls(int maxTotalCalls) {
            this.maxTotalCalls = maxTotalCalls;
            return this;
        }

        public ToolCallGuardrailAdvisor build() {
            return new ToolCallGuardrailAdvisor(order, allowedTools, dangerousTools,
                    maxCallsPerTool, maxTotalCalls);
        }
    }
}
