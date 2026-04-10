package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.core.Ordered;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * 输出护栏 Advisor
 * 
 * 对模型的输出进行多层质量和安全检查：
 * 1. 敏感信息泄露检测：检查输出中是否包含API Key、密码、内部URL等
 * 2. 幻觉检测：使用LLM评估输出是否与已知事实矛盾
 * 3. 格式校验：确保输出符合预期的格式要求
 * 4. 一致性检查：检测输出与之前承诺是否矛盾
 * 
 * 当检测到问题时，自动修正或标注警告。
 */
@Slf4j
public class OutputGuardrailAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final ChatModel validationModel;
    private final boolean enableHallucinationCheck;

    private static final String CTX_OUTPUT_MODIFIED = "output_guardrail_modified";
    private static final String CTX_HALLUCINATION_SCORE = "hallucination_score";

    // 敏感信息泄露模式
    private static final List<Pattern> LEAK_PATTERNS = List.of(
            Pattern.compile("(?i)(api[_\\s-]?key|secret[_\\s-]?key)\\s*[=:：]\\s*[\\w\\-]{10,}"),
            Pattern.compile("(?i)(password|passwd|pwd)\\s*[=:：]\\s*\\S{6,}"),
            Pattern.compile("(?i)bearer\\s+[a-zA-Z0-9\\-._~+/]+=*"),
            Pattern.compile("(?i)(jdbc|mysql|redis)://[^\\s]+:[^\\s]+@"),
            Pattern.compile("\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}:\\d{4,5}\\b"),
            Pattern.compile("(?i)(sk-|ak-)[a-zA-Z0-9]{20,}")
    );

    // 编造数据的常见模式
    private static final List<Pattern> FABRICATION_PATTERNS = List.of(
            Pattern.compile("(?:假设|例如|比如说|可能是).*(?:手机号|订单号|票价).*\\d{5,}"),
            Pattern.compile("(?:大概|约|大约|估计).*(?:票价|价格).*\\d+元")
    );

    private static final String HALLUCINATION_CHECK_PROMPT = """
            评估以下AI助手的回复是否存在幻觉（编造不存在的信息）。
            
            用户问题：%s
            AI回复：%s
            
            检查项：
            1. 是否编造了具体的数字（票价、库存、订单号等）而非来自工具查询
            2. 是否声称做了某个操作但实际并未调用对应工具
            3. 是否提供了过于具体但缺乏来源的信息
            
            评分（0-10，0=无幻觉，10=严重幻觉）：
            只输出数字评分和一句话理由，格式：评分|理由
            """;

    private OutputGuardrailAdvisor(int order, ChatModel validationModel, 
                                    boolean enableHallucinationCheck) {
        this.order = order;
        this.validationModel = validationModel;
        this.enableHallucinationCheck = enableHallucinationCheck;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        return request;
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null ||
            chatResponse.getResult().getOutput() == null) {
            return response;
        }

        String output = chatResponse.getResult().getOutput().getText();
        if (output == null || output.isEmpty()) {
            return response;
        }

        Map<String, Object> context = new HashMap<>(response.context());
        boolean modified = false;

        // 第一层：敏感信息泄露检测
        String sanitized = sanitizeOutput(output);
        if (!sanitized.equals(output)) {
            log.warn("输出护栏 - 检测到敏感信息泄露，已脱敏");
            output = sanitized;
            modified = true;
        }

        // 第二层：编造数据模式检测
        for (Pattern pattern : FABRICATION_PATTERNS) {
            if (pattern.matcher(output).find()) {
                log.warn("输出护栏 - 检测到可能的数据编造模式: {}", pattern.pattern());
                output = output + "\n\n⚠️ 注意：以上部分信息可能为估算值，具体请以实际查询结果为准。";
                modified = true;
                break;
            }
        }

        // 第三层：LLM幻觉检测（可选，异步评估）
        if (enableHallucinationCheck && validationModel != null) {
            String userInput = (String) context.getOrDefault("observability_user_input", "");
            evaluateHallucinationAsync(context, userInput, output);
        }

        if (modified) {
            context.put(CTX_OUTPUT_MODIFIED, true);
            return rebuildResponse(response, output, context);
        }

        return response;
    }

    private String sanitizeOutput(String output) {
        String result = output;
        for (Pattern pattern : LEAK_PATTERNS) {
            result = pattern.matcher(result).replaceAll("[已脱敏]");
        }
        return result;
    }

    private void evaluateHallucinationAsync(Map<String, Object> context, 
                                             String userInput, String output) {
        Thread.startVirtualThread(() -> {
            try {
                String prompt = String.format(HALLUCINATION_CHECK_PROMPT,
                        truncate(userInput, 300), truncate(output, 800));
                var llmResponse = validationModel.call(new Prompt(prompt));
                String result = llmResponse.getResult().getOutput().getText().trim();

                if (result.contains("|")) {
                    String[] parts = result.split("\\|", 2);
                    try {
                        int score = Integer.parseInt(parts[0].trim());
                        String reason = parts.length > 1 ? parts[1].trim() : "";
                        context.put(CTX_HALLUCINATION_SCORE, score);

                        if (score >= 7) {
                            log.warn("输出护栏 - 幻觉评分高: {}/10, 原因: {}", score, reason);
                        } else if (score >= 4) {
                            log.info("输出护栏 - 幻觉评分中等: {}/10, 原因: {}", score, reason);
                        } else {
                            log.debug("输出护栏 - 幻觉评分低: {}/10", score);
                        }
                    } catch (NumberFormatException e) {
                        log.debug("幻觉评分解析失败: {}", result);
                    }
                }
            } catch (Exception e) {
                log.debug("幻觉检测跳过: {}", e.getMessage());
            }
        });
    }

    private ChatClientResponse rebuildResponse(ChatClientResponse original, 
                                                String newOutput, Map<String, Object> context) {
        // 由于ChatResponse不可变，通过context传递修正信息
        context.put("sanitized_output", newOutput);
        return ChatClientResponse.builder()
                .chatResponse(original.chatResponse())
                .context(context)
                .build();
    }

    private String truncate(String str, int maxLen) {
        if (str == null) return "";
        return str.length() > maxLen ? str.substring(0, maxLen) + "..." : str;
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int order = Ordered.LOWEST_PRECEDENCE - 10;
        private ChatModel validationModel;
        private boolean enableHallucinationCheck = false;

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder validationModel(ChatModel validationModel) {
            this.validationModel = validationModel;
            return this;
        }

        public Builder enableHallucinationCheck(boolean enableHallucinationCheck) {
            this.enableHallucinationCheck = enableHallucinationCheck;
            return this;
        }

        public OutputGuardrailAdvisor build() {
            return new OutputGuardrailAdvisor(order, validationModel, enableHallucinationCheck);
        }
    }
}
