package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.core.Ordered;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 熔断器 Advisor (Guardrails 工程)
 *
 * 借鉴微服务的 Circuit Breaker 模式，对 LLM 调用和工具调用实施熔断保护：
 *
 * 三种状态：
 * - CLOSED（正常）：请求正常通过，记录失败次数
 * - OPEN（熔断）：连续失败达到阈值，直接返回降级响应，不再调用LLM/工具
 * - HALF_OPEN（半开）：熔断冷却期过后，放行一个探测请求，成功则恢复，失败则继续熔断
 *
 * 降级策略：
 * - LLM 熔断时：返回友好的"服务繁忙"提示 + 建议稍后重试
 * - 工具熔断时：在上下文注入"该工具暂不可用"，让模型跳过该步骤
 *
 * 应用场景：
 * - LLM API 频繁超时/5xx → 避免用户长时间等待
 * - MCP 工具连接不稳定 → 快速失败，避免 Plan-Execute 陷入死循环
 */
@Slf4j
public class CircuitBreakerAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final int failureThreshold;
    private final long cooldownMillis;

    // 全局熔断状态（跨会话共享，单例安全）
    private static final ConcurrentHashMap<String, CircuitState> CIRCUITS = new ConcurrentHashMap<>();

    private static final String CTX_CIRCUIT_STATUS = "circuit_breaker_status";
    private static final String CIRCUIT_LLM = "llm_call";

    private static final String DEGRADED_RESPONSE = """
            😔 非常抱歉，当前AI服务响应较慢，可能正在经历高峰期。
            
            您可以：
            1. **稍后重试** - 通常几分钟后会恢复正常
            2. **简化问题** - 尝试用更简短的方式描述您的需求
            3. **联系人工客服** - 如果问题紧急，建议联系人工服务
            
            给您带来不便，深感抱歉！🙏
            """;

    private CircuitBreakerAdvisor(int order, int failureThreshold, long cooldownMillis) {
        this.order = order;
        this.failureThreshold = failureThreshold;
        this.cooldownMillis = cooldownMillis;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        CircuitState state = CIRCUITS.computeIfAbsent(CIRCUIT_LLM,
                k -> new CircuitState(failureThreshold, cooldownMillis));

        Map<String, Object> context = new HashMap<>(request.context());

        switch (state.getStatus()) {
            case OPEN:
                // 检查是否可以进入半开状态
                if (state.shouldAttemptReset()) {
                    state.transitionToHalfOpen();
                    context.put(CTX_CIRCUIT_STATUS, "HALF_OPEN");
                    log.info("CircuitBreaker: 冷却期结束，进入HALF_OPEN状态，允许一次探测请求");
                } else {
                    context.put(CTX_CIRCUIT_STATUS, "OPEN");
                    log.warn("CircuitBreaker: 熔断中，剩余冷却时间 {}秒",
                            (state.cooldownEndTime.get() - Instant.now().toEpochMilli()) / 1000);
                    // 不再继续 chain，直接返回（在 after 中生成降级响应）
                }
                break;

            case HALF_OPEN:
                context.put(CTX_CIRCUIT_STATUS, "HALF_OPEN");
                log.info("CircuitBreaker: HALF_OPEN状态，探测请求通过");
                break;

            case CLOSED:
                context.put(CTX_CIRCUIT_STATUS, "CLOSED");
                break;
        }

        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        Map<String, Object> context = new HashMap<>(response.context());
        String status = (String) context.getOrDefault(CTX_CIRCUIT_STATUS, "CLOSED");
        CircuitState state = CIRCUITS.computeIfAbsent(CIRCUIT_LLM,
                k -> new CircuitState(failureThreshold, cooldownMillis));

        ChatResponse chatResponse = response.chatResponse();

        // 熔断状态下直接返回降级响应
        if ("OPEN".equals(status)) {
            log.warn("CircuitBreaker: 返回降级响应");
            ChatResponse degraded = buildDegradedResponse();
            return ChatClientResponse.builder().chatResponse(degraded).context(context).build();
        }

        // 检查响应是否成功
        boolean isSuccess = chatResponse != null
                && chatResponse.getResult() != null
                && chatResponse.getResult().getOutput() != null
                && chatResponse.getResult().getOutput().getText() != null
                && !chatResponse.getResult().getOutput().getText().isBlank();

        boolean hasError = false;
        if (isSuccess) {
            String output = chatResponse.getResult().getOutput().getText();
            // 检测是否是错误响应（LLM返回了错误信息）
            hasError = output.contains("Internal Server Error")
                    || output.contains("rate_limit_exceeded")
                    || output.contains("model_overloaded");
        }

        if (isSuccess && !hasError) {
            state.recordSuccess();
            if ("HALF_OPEN".equals(status)) {
                log.info("CircuitBreaker: 探测请求成功，恢复CLOSED状态");
            }
        } else {
            state.recordFailure();
            if (state.getStatus() == BreakerStatus.OPEN) {
                log.error("CircuitBreaker: 连续失败 {} 次，触发熔断！冷却 {}秒",
                        failureThreshold, cooldownMillis / 1000);
            }
        }

        return response;
    }

    private ChatResponse buildDegradedResponse() {
        AssistantMessage msg = new AssistantMessage(DEGRADED_RESPONSE);
        Generation gen = new Generation(msg);
        return new ChatResponse(List.of(gen));
    }

    @Override
    public int getOrder() { return order; }

    public static Builder builder() { return new Builder(); }

    // 重置所有熔断器（用于测试或手动恢复）
    public static void resetAll() {
        CIRCUITS.clear();
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 10;
        private int failureThreshold = 5;
        private long cooldownMillis = 60_000L;

        public Builder order(int order) { this.order = order; return this; }
        public Builder failureThreshold(int threshold) { this.failureThreshold = threshold; return this; }
        public Builder cooldownSeconds(int seconds) { this.cooldownMillis = seconds * 1000L; return this; }

        public CircuitBreakerAdvisor build() {
            return new CircuitBreakerAdvisor(order, failureThreshold, cooldownMillis);
        }
    }

    // ─────────── 熔断器状态机 ───────────

    private enum BreakerStatus { CLOSED, OPEN, HALF_OPEN }

    private static class CircuitState {
        private volatile BreakerStatus status = BreakerStatus.CLOSED;
        private final AtomicInteger consecutiveFailures = new AtomicInteger(0);
        private final AtomicLong cooldownEndTime = new AtomicLong(0);
        private final int threshold;
        private final long cooldownMs;

        CircuitState(int threshold, long cooldownMs) {
            this.threshold = threshold;
            this.cooldownMs = cooldownMs;
        }

        synchronized BreakerStatus getStatus() { return status; }

        synchronized void recordSuccess() {
            consecutiveFailures.set(0);
            status = BreakerStatus.CLOSED;
        }

        synchronized void recordFailure() {
            int failures = consecutiveFailures.incrementAndGet();
            if (failures >= threshold && status != BreakerStatus.OPEN) {
                status = BreakerStatus.OPEN;
                cooldownEndTime.set(Instant.now().toEpochMilli() + cooldownMs);
            }
        }

        boolean shouldAttemptReset() {
            return status == BreakerStatus.OPEN
                    && Instant.now().toEpochMilli() >= cooldownEndTime.get();
        }

        synchronized void transitionToHalfOpen() {
            status = BreakerStatus.HALF_OPEN;
        }
    }
}
