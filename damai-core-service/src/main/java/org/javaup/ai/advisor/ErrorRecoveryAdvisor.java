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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 运维 Agent 工具失败自恢复 Advisor
 *
 * 从 cc-haha-main (Claude Code) 的工程实践中借鉴的设计思想：
 * - 分析工具调用的错误输出，按错误类型选择不同恢复策略
 * - 参数错误时自动注入修正指引（类似 Claude Code 的 self-fix 能力）
 * - Token 预算感知：连续无进展时注入"收尾指令"（借鉴 tokenBudget.ts 的边际收益递减检测）
 * - 防循环保护：同一错误类型最多恢复 maxRecoveryPerType 次
 *
 * 错误类型 → 恢复策略矩阵：
 *   TIMEOUT         → 指数退避重试，缩小查询范围
 *   SERVICE_DOWN    → 切换备用工具，跳过该步骤继续
 *   NO_RESULT       → 扩大时间范围，降低过滤条件
 *   PARAM_ERROR     → LLM 自动分析并修正参数格式
 *   PERMISSION_DENIED → 降级只读，标记数据缺失
 */
@Slf4j
public class ErrorRecoveryAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final int maxRecoveryPerType;
    private final int tokenBudgetWarningThreshold;

    private static final String CTX_RECOVERY_COUNTS   = "error_recovery_counts";
    private static final String CTX_LAST_TOKEN_USAGE  = "error_recovery_last_tokens";
    private static final String CTX_LOW_PROGRESS_CNT  = "error_recovery_low_progress_count";
    private static final String CTX_RECOVERY_INJECTED = "error_recovery_injected_this_turn";

    private static final int DIMINISHING_TOKEN_DELTA  = 300;
    private static final int DIMINISHING_COUNT_LIMIT  = 3;

    private ErrorRecoveryAdvisor(int order, int maxRecoveryPerType, int tokenBudgetWarningThreshold) {
        this.order = order;
        this.maxRecoveryPerType = maxRecoveryPerType;
        this.tokenBudgetWarningThreshold = tokenBudgetWarningThreshold;
    }

    // ─────────────────── before ───────────────────

    @Override
    @SuppressWarnings("unchecked")
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        Map<String, Object> context = new HashMap<>(request.context());

        context.putIfAbsent(CTX_RECOVERY_COUNTS, new HashMap<String, Integer>());
        context.putIfAbsent(CTX_LAST_TOKEN_USAGE, 0L);
        context.putIfAbsent(CTX_LOW_PROGRESS_CNT, 0);
        context.remove(CTX_RECOVERY_INJECTED);

        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }

    // ─────────────────── after ────────────────────

    @Override
    @SuppressWarnings("unchecked")
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }

        Map<String, Object> context = new HashMap<>(response.context());
        Map<String, Integer> recoveryCounts =
                (Map<String, Integer>) context.computeIfAbsent(CTX_RECOVERY_COUNTS, k -> new HashMap<>());

        String output = chatResponse.getResult().getOutput().getText();
        List<ToolCall> toolCalls = chatResponse.getResult().getToolCalls();

        List<Message> injections = new ArrayList<>();

        // ── 1. 检测工具调用错误，按类型注入恢复指令 ──
        if (toolCalls != null && !toolCalls.isEmpty()) {
            ErrorType errorType = detectErrorType(output);
            if (errorType != null) {
                int currentCount = recoveryCounts.getOrDefault(errorType.name(), 0);
                if (currentCount < maxRecoveryPerType) {
                    String instruction = buildRecoveryInstruction(errorType, output);
                    injections.add(new SystemMessage(instruction));
                    recoveryCounts.put(errorType.name(), currentCount + 1);
                    context.put(CTX_RECOVERY_INJECTED, true);
                    log.warn("ErrorRecovery: 检测到 [{}] 错误 (第{}/{}次恢复)，注入恢复指令",
                            errorType, currentCount + 1, maxRecoveryPerType);
                } else {
                    // 超过最大恢复次数，注入放弃指令
                    injections.add(new SystemMessage(String.format("""
                            【跳过失败步骤】
                            工具 [%s] 已连续失败 %d 次，无法继续恢复。
                            请记录该数据维度缺失，继续执行下一个诊断步骤。
                            在最终报告中注明："%s 维度数据获取失败，结论基于现有数据。"
                            """, extractFailedToolName(toolCalls), maxRecoveryPerType, errorType.description)));
                    log.warn("ErrorRecovery: [{}] 已达最大恢复次数 {}，注入跳过指令", errorType, maxRecoveryPerType);
                }
            }
        }

        // ── 2. Token 预算感知：边际收益递减检测（借鉴 cc-haha-main tokenBudget.ts）──
        if (tokenBudgetWarningThreshold > 0 && chatResponse.getMetadata() != null
                && chatResponse.getMetadata().getUsage() != null) {
            long totalTokens = chatResponse.getMetadata().getUsage().getTotalTokens();
            long lastTokens  = (Long) context.getOrDefault(CTX_LAST_TOKEN_USAGE, 0L);
            int  lowProgCnt  = (Integer) context.getOrDefault(CTX_LOW_PROGRESS_CNT, 0);

            long delta = totalTokens - lastTokens;

            if (totalTokens > tokenBudgetWarningThreshold) {
                if (delta < DIMINISHING_TOKEN_DELTA) {
                    lowProgCnt++;
                    log.debug("ErrorRecovery: Token进展低迷，本轮增量={}, 连续低迷次数={}", delta, lowProgCnt);
                    if (lowProgCnt >= DIMINISHING_COUNT_LIMIT) {
                        injections.add(new SystemMessage(String.format("""
                                【Token预算提醒】
                                已消耗约 %d tokens，且连续 %d 轮进展较少。
                                请基于已收集到的信息立即给出诊断结论，不要继续查询新数据。
                                """, totalTokens, lowProgCnt)));
                        log.warn("ErrorRecovery: 边际收益递减（连续{}轮），注入收尾指令", lowProgCnt);
                    }
                } else {
                    lowProgCnt = 0;
                }
            }

            context.put(CTX_LAST_TOKEN_USAGE, totalTokens);
            context.put(CTX_LOW_PROGRESS_CNT, lowProgCnt);
        }

        // ── 3. 将注入内容追加到消息列表 ──
        if (!injections.isEmpty()) {
            // 借助 AdvisorChain 的 context 在下一轮 before() 时由其他 Advisor 读取
            // 这里通过将消息暂存到 context 传递给下一轮的 before 阶段处理
            context.put("error_recovery_pending_messages", injections);
        }

        context.put(CTX_RECOVERY_COUNTS, recoveryCounts);
        return ChatClientResponse.builder()
                .chatResponse(chatResponse)
                .context(context)
                .build();
    }

    // ─────────────────── 错误检测 ────────────────────

    enum ErrorType {
        TIMEOUT("工具调用超时"),
        SERVICE_DOWN("依赖服务不可用"),
        NO_RESULT("查询无结果"),
        PARAM_ERROR("参数格式错误"),
        PERMISSION_DENIED("权限不足");

        final String description;
        ErrorType(String description) { this.description = description; }
    }

    private ErrorType detectErrorType(String output) {
        if (output == null) return null;
        String lower = output.toLowerCase();

        if (lower.contains("timeout") || lower.contains("超时") || lower.contains("timed out")) {
            return ErrorType.TIMEOUT;
        }
        if (lower.contains("503") || lower.contains("unavailable") || lower.contains("cluster")
                || lower.contains("连接拒绝") || lower.contains("connection refused")) {
            return ErrorType.SERVICE_DOWN;
        }
        if (lower.contains("no result") || lower.contains("未找到") || lower.contains("0 条")
                || lower.contains("0条") || lower.contains("没有找到") || lower.contains("empty result")) {
            return ErrorType.NO_RESULT;
        }
        if (lower.contains("invalid") || lower.contains("格式错误") || lower.contains("format error")
                || lower.contains("parse error") || lower.contains("bad request") || lower.contains("400")) {
            return ErrorType.PARAM_ERROR;
        }
        if (lower.contains("403") || lower.contains("permission") || lower.contains("forbidden")
                || lower.contains("无权限") || lower.contains("access denied")) {
            return ErrorType.PERMISSION_DENIED;
        }
        // 通用失败检测
        if (lower.contains("error") || lower.contains("exception") || lower.contains("失败")) {
            return ErrorType.SERVICE_DOWN;
        }
        return null;
    }

    private String buildRecoveryInstruction(ErrorType type, String errorOutput) {
        String errMsg = truncate(errorOutput, 300);
        return switch (type) {
            case TIMEOUT -> String.format("""
                    【自动恢复 - 工具超时】
                    错误信息：%s
                    
                    恢复策略（按优先级执行）：
                    1. 用相同参数重试一次（网络抖动可能导致偶发超时）
                    2. 若仍超时：将时间范围缩短至 1/4（如查1小时改为查15分钟）
                    3. 若仍超时：跳过此步骤，在报告中注明"日志查询超时，该时间段数据缺失"
                    """, errMsg);

            case SERVICE_DOWN -> String.format("""
                    【自动恢复 - 服务不可用】
                    错误信息：%s
                    
                    恢复策略（按优先级执行）：
                    1. 检查是否有备用工具可替代（如 backup_log_query、metrics_query）
                    2. 若无备用工具：跳过此步骤，改为从其他维度收集信息（链路追踪/指标监控）
                    3. 在最终报告中注明"[工具名]不可用，日志维度数据缺失"
                    """, errMsg);

            case NO_RESULT -> String.format("""
                    【自动恢复 - 查询无结果】
                    错误信息：%s
                    
                    恢复策略（按优先级执行）：
                    1. 扩大时间范围：将查询窗口扩大到原来的 4 倍
                    2. 降低过滤条件：去掉 level=ERROR 限制，查询所有级别日志
                    3. 检查服务名/实例名拼写（大小写、连字符格式）
                    4. 若仍无结果：记录"该时间段内无相关日志"，继续下一步
                    """, errMsg);

            case PARAM_ERROR -> String.format("""
                    【自动修正 - 参数格式错误】
                    错误信息：%s
                    
                    请按以下步骤自动修正并重试：
                    1. 分析错误信息，明确哪个参数格式不对
                    2. 常见修正规则：
                       - traceId：应为 32 位十六进制字符串（无连字符）
                       - 时间格式：应为 ISO 8601（如 2024-01-01T10:00:00Z）
                       - 服务名：通常为小写 + 连字符（如 order-service）
                    3. 说明"原参数"→"修正后参数"，然后重新调用工具
                    """, errMsg);

            case PERMISSION_DENIED -> String.format("""
                    【自动恢复 - 权限不足】
                    错误信息：%s
                    
                    恢复策略：
                    1. 改用只读查询接口（去掉需要写权限的操作）
                    2. 不要重试相同调用（权限问题重试无意义）
                    3. 在报告中注明"受权限限制，以下数据无法获取：[具体数据]"
                    4. 基于已有的其他维度数据，尽可能给出诊断结论
                    """, errMsg);
        };
    }

    private String extractFailedToolName(List<ToolCall> toolCalls) {
        if (toolCalls == null || toolCalls.isEmpty()) return "unknown";
        return toolCalls.stream().map(ToolCall::name).findFirst().orElse("unknown");
    }

    private String truncate(String s, int maxLen) {
        if (s == null) return "";
        return s.length() > maxLen ? s.substring(0, maxLen) + "..." : s;
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 200;
        private int maxRecoveryPerType = 2;
        private int tokenBudgetWarningThreshold = 8000;

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder maxRecoveryPerType(int max) {
            this.maxRecoveryPerType = max;
            return this;
        }

        public Builder tokenBudgetWarningThreshold(int threshold) {
            this.tokenBudgetWarningThreshold = threshold;
            return this;
        }

        public ErrorRecoveryAdvisor build() {
            return new ErrorRecoveryAdvisor(order, maxRecoveryPerType, tokenBudgetWarningThreshold);
        }
    }
}
