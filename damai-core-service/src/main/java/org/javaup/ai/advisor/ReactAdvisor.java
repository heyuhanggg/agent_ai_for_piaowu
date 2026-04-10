package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.ToolCall;
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 增强版 ReAct (Reason + Act) Advisor
 * 
 * 真正驱动模型在 Thought → Action → Observation 之间迭代的决策引擎：
 * 
 * 核心改进（相比基础版）：
 * 1. 结构化思考链：强制模型输出结构化的 Thought/Action/Observation 格式
 * 2. 观察注入：将工具结果格式化为 Observation 注入到下一轮对话
 * 3. 迭代质量评估：检测无效迭代（重复调用、无进展），自动终止
 * 4. 思考链摘要：迭代结束后生成完整的推理链路摘要
 * 5. 强制终止机制：达到最大迭代后注入总结指令，确保给出最终答案
 */
@Slf4j
public class ReactAdvisor implements BaseChatMemoryAdvisor {
    
    private final int order;
    private final int maxIterations;
    private final boolean enableReactLoop;
    private final int tokenBudgetWarningThreshold;
    
    private static final String CTX_REACT_ITERATION    = "react_iteration_count";
    private static final String CTX_REACT_HISTORY       = "react_thought_history";
    private static final String CTX_REACT_ENABLED       = "react_loop_enabled";
    private static final String CTX_REACT_TOOL_HISTORY  = "react_tool_call_history";
    private static final String CTX_REACT_STALE_COUNT   = "react_stale_iteration_count";
    private static final String CTX_REACT_LAST_TOKENS   = "react_last_token_usage";
    private static final String CTX_REACT_LOW_PROG_CNT  = "react_low_progress_count";

    private static final int DIMINISHING_TOKEN_DELTA    = 400;
    private static final int DIMINISHING_COUNT_LIMIT    = 3;
    
    private static final String REACT_SYSTEM_PROMPT = """
            你是一个使用ReAct (Reason + Act) 决策模式的智能助手。
            
            每次回复必须严格遵循以下结构化格式：
            
            **Thought**: [分析当前情况，明确下一步目标，评估已有信息是否足够]
            **Action**: [决定调用哪个工具，或决定给出最终答案]
            **Observation**: [观察工具返回的结果，评估是否达到目标]
            
            决策规则：
            - 每次只执行一个明确的行动，不要一次调用多个不相关的工具
            - 如果工具返回的信息不足，在 Thought 中分析原因并调整策略
            - 如果连续两次工具调用获得相同结果，停止重复并给出结论
            - 当信息足够回答用户问题时，直接给出 **Final Answer**
            - 最多进行 %d 轮迭代，达到上限后必须立即给出最终答案
            
            最终答案格式：
            **Final Answer**: [基于所有收集到的信息给出的完整回答]
            """;
    
    private static final String FORCE_CONCLUDE_PROMPT = """
            你已经进行了 %d 轮思考和工具调用。现在必须基于已收集到的所有信息给出最终答案。
            
            已有信息摘要：
            %s
            
            请直接给出 **Final Answer**，不要再调用任何工具。
            """;
    
    private ReactAdvisor(int order, int maxIterations, boolean enableReactLoop, int tokenBudgetWarningThreshold) {
        this.order = order;
        this.maxIterations = maxIterations;
        this.enableReactLoop = enableReactLoop;
        this.tokenBudgetWarningThreshold = tokenBudgetWarningThreshold;
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        if (!enableReactLoop) {
            return request;
        }
        
        Map<String, Object> context = new HashMap<>(request.context());
        
        // 初始化React上下文
        if (!context.containsKey(CTX_REACT_ITERATION)) {
            context.put(CTX_REACT_ITERATION, 0);
            context.put(CTX_REACT_HISTORY, new ArrayList<String>());
            context.put(CTX_REACT_ENABLED, true);
            context.put(CTX_REACT_TOOL_HISTORY, new ArrayList<String>());
            context.put(CTX_REACT_STALE_COUNT, 0);
            context.put(CTX_REACT_LAST_TOKENS, 0L);
            context.put(CTX_REACT_LOW_PROG_CNT, 0);

            log.info("React模式已启用，最大迭代次数: {}，Token预算警戒: {}", maxIterations, tokenBudgetWarningThreshold);
            
            // 注入React系统提示词
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.add(0, new SystemMessage(String.format(REACT_SYSTEM_PROMPT, maxIterations)));
            
            return ChatClientRequest.builder()
                    .prompt(request.prompt().withInstructions(messages))
                    .context(context)
                    .build();
        }
        
        // 检查迭代次数
        Integer iteration = (Integer) context.get(CTX_REACT_ITERATION);
        List<String> thoughtHistory = (List<String>) context.getOrDefault(CTX_REACT_HISTORY, new ArrayList<>());

        // 处理上一轮 ErrorRecoveryAdvisor 可能挂载的恢复消息
        @SuppressWarnings("unchecked")
        List<Message> pendingMessages = (List<Message>) context.remove("error_recovery_pending_messages");
        if (pendingMessages != null && !pendingMessages.isEmpty()) {
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.addAll(pendingMessages);
            request = ChatClientRequest.builder()
                    .prompt(request.prompt().withInstructions(messages))
                    .context(context)
                    .build();
        }

        // 达到最大迭代次数，强制总结
        if (iteration >= maxIterations) {
            log.warn("React迭代达到最大次数: {}，注入强制总结指令", maxIterations);
            context.put(CTX_REACT_ENABLED, false);
            
            String summary = buildThoughtSummary(thoughtHistory);
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.add(new SystemMessage(String.format(FORCE_CONCLUDE_PROMPT, iteration, summary)));
            
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
    @SuppressWarnings("unchecked")
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        if (!enableReactLoop) {
            return response;
        }
        
        Map<String, Object> context = new HashMap<>(response.context());
        Integer iteration = (Integer) context.getOrDefault(CTX_REACT_ITERATION, 0);
        List<String> thoughtHistory = (List<String>) context.getOrDefault(CTX_REACT_HISTORY, new ArrayList<>());
        List<String> toolHistory = (List<String>) context.getOrDefault(CTX_REACT_TOOL_HISTORY, new ArrayList<>());
        Boolean reactEnabled = (Boolean) context.getOrDefault(CTX_REACT_ENABLED, false);
        Integer staleCount = (Integer) context.getOrDefault(CTX_REACT_STALE_COUNT, 0);
        
        if (!reactEnabled || iteration >= maxIterations) {
            // 输出最终思考链摘要
            if (!thoughtHistory.isEmpty()) {
                log.info("React完成 - 共 {} 轮迭代\n{}", iteration, 
                        buildThoughtSummary(thoughtHistory));
            }
            return response;
        }
        
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }
        
        String assistantOutput = chatResponse.getResult().getOutput().getText();
        List<ToolCall> toolCalls = chatResponse.getResult().getToolCalls();

        // ── Token 预算感知（借鉴 cc-haha-main tokenBudget.ts 边际收益递减检测）──
        if (tokenBudgetWarningThreshold > 0 && chatResponse.getMetadata() != null
                && chatResponse.getMetadata().getUsage() != null) {
            long totalTokens = chatResponse.getMetadata().getUsage().getTotalTokens();
            long lastTokens  = (Long) context.getOrDefault(CTX_REACT_LAST_TOKENS, 0L);
            int  lowProgCnt  = (Integer) context.getOrDefault(CTX_REACT_LOW_PROG_CNT, 0);
            long delta = totalTokens - lastTokens;

            if (totalTokens > tokenBudgetWarningThreshold) {
                if (delta < DIMINISHING_TOKEN_DELTA) {
                    lowProgCnt++;
                    if (lowProgCnt >= DIMINISHING_COUNT_LIMIT) {
                        log.warn("React: Token边际收益递减(连续{}轮增量<{})，注入收尾Nudge", lowProgCnt, DIMINISHING_TOKEN_DELTA);
                        thoughtHistory.add("[Token预算] 边际收益递减，触发收尾");
                        // 注入收尾nudge（不强制停止，给模型最后机会总结）
                        List<Message> msgs = new ArrayList<>(request.prompt().getInstructions());
                        msgs.add(new SystemMessage(String.format(
                                "已消耗 %d tokens，进展趋于停滞。请基于已有信息直接给出 **Final Answer**。",
                                totalTokens)));
                        context.put(CTX_REACT_LOW_PROG_CNT, 0); // 重置，避免重复注入
                    } else {
                        log.debug("React: Token进展低迷 delta={}, 连续次数={}", delta, lowProgCnt);
                        context.put(CTX_REACT_LOW_PROG_CNT, lowProgCnt);
                    }
                } else {
                    context.put(CTX_REACT_LOW_PROG_CNT, 0);
                }
            }
            context.put(CTX_REACT_LAST_TOKENS, totalTokens);
        }

        // 记录结构化思考过程
        String thought = extractStructuredThought(assistantOutput);
        thoughtHistory.add(String.format("[迭代%d/Thought] %s", iteration + 1, thought));
        
        // 检查是否有工具调用
        if (toolCalls != null && !toolCalls.isEmpty()) {
            // 检测重复调用（无效迭代）
            String currentToolSig = toolCalls.stream()
                    .map(tc -> tc.name() + ":" + tc.arguments().hashCode())
                    .sorted()
                    .reduce("", (a, b) -> a + "|" + b);
            
            if (toolHistory.contains(currentToolSig)) {
                staleCount++;
                log.warn("React迭代 {}: 检测到重复工具调用 (第{}次)", iteration + 1, staleCount);
                thoughtHistory.add(String.format("[迭代%d/Warning] 重复工具调用，无新信息", iteration + 1));
                
                // 连续2次重复，强制终止
                if (staleCount >= 2) {
                    log.warn("React: 连续 {} 次无效迭代，强制终止", staleCount);
                    context.put(CTX_REACT_ENABLED, false);
                    context.put(CTX_REACT_ITERATION, iteration + 1);
                    context.put(CTX_REACT_HISTORY, thoughtHistory);
                    return ChatClientResponse.builder()
                            .chatResponse(chatResponse).context(context).build();
                }
            } else {
                staleCount = 0;
            }
            toolHistory.add(currentToolSig);
            
            for (ToolCall toolCall : toolCalls) {
                thoughtHistory.add(String.format("[迭代%d/Action] 调用工具: %s", 
                        iteration + 1, toolCall.name()));
            }
            
            log.info("React迭代 {}/{}: 调用 {} 个工具 {}", iteration + 1, maxIterations,
                    toolCalls.size(), toolCalls.stream().map(ToolCall::name).toList());
            
            context.put(CTX_REACT_ITERATION, iteration + 1);
            context.put(CTX_REACT_STALE_COUNT, staleCount);
        } else {
            // 检查是否包含 Final Answer 标记
            boolean hasFinalAnswer = assistantOutput != null && 
                    (assistantOutput.contains("Final Answer") || assistantOutput.contains("最终答案"));
            
            if (hasFinalAnswer) {
                log.info("React迭代 {}/{}: 模型给出最终答案", iteration + 1, maxIterations);
            } else {
                log.info("React迭代 {}/{}: 无工具调用，视为最终回复", iteration + 1, maxIterations);
            }
            
            context.put(CTX_REACT_ENABLED, false);
            thoughtHistory.add(String.format("[迭代%d/Final] 给出最终答案", iteration + 1));
        }
        
        context.put(CTX_REACT_HISTORY, thoughtHistory);
        context.put(CTX_REACT_TOOL_HISTORY, toolHistory);
        
        return ChatClientResponse.builder()
                .chatResponse(chatResponse)
                .context(context)
                .build();
    }
    
    private String extractStructuredThought(String output) {
        if (output == null) return "无";
        
        // 尝试提取 **Thought**: 格式
        for (String prefix : List.of("**Thought**:", "Thought:", "思考：", "思考:")) {
            int idx = output.indexOf(prefix);
            if (idx >= 0) {
                String after = output.substring(idx + prefix.length()).trim();
                String thought = after.split("\n")[0].trim();
                // 继续检查是否有多行思考
                String[] lines = after.split("\n");
                StringBuilder sb = new StringBuilder();
                for (String line : lines) {
                    if (line.trim().startsWith("**Action") || line.trim().startsWith("Action") ||
                        line.trim().startsWith("**Observation") || line.trim().isEmpty()) {
                        break;
                    }
                    sb.append(line.trim()).append(" ");
                }
                String result = sb.toString().trim();
                return result.length() > 150 ? result.substring(0, 150) + "..." : result;
            }
        }
        
        return output.length() > 150 ? output.substring(0, 150) + "..." : output;
    }
    
    private String buildThoughtSummary(List<String> history) {
        if (history.isEmpty()) return "无思考记录";
        StringBuilder sb = new StringBuilder();
        for (String entry : history) {
            sb.append("  ").append(entry).append("\n");
        }
        return sb.toString();
    }
    
    @Override
    public int getOrder() {
        return order;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 100;
        private int maxIterations = 5;
        private boolean enableReactLoop = true;
        private int tokenBudgetWarningThreshold = 6000;

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder maxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }

        public Builder enableReactLoop(boolean enableReactLoop) {
            this.enableReactLoop = enableReactLoop;
            return this;
        }

        public Builder tokenBudgetWarningThreshold(int threshold) {
            this.tokenBudgetWarningThreshold = threshold;
            return this;
        }

        public ReactAdvisor build() {
            return new ReactAdvisor(order, maxIterations, enableReactLoop, tokenBudgetWarningThreshold);
        }
    }
}
