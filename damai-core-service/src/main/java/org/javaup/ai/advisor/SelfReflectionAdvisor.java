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
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 自我反思 Advisor
 * 
 * 实现 Self-Reflection 机制，让模型在给出最终回复前先进行自我评估：
 * 1. 完整性检查：回复是否完整回答了用户的问题
 * 2. 准确性检查：回复中的信息是否有依据（工具结果或已知事实）
 * 3. 安全性检查：回复是否符合安全规范
 * 4. 可操作性检查：给出的建议是否可以实际执行
 * 
 * 如果反思评分低于阈值，会注入改进指令让模型重新生成。
 */
@Slf4j
public class SelfReflectionAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final ChatModel reflectionModel;
    private final int qualityThreshold;

    private static final String CTX_REFLECTION_SCORE = "self_reflection_score";
    private static final String CTX_REFLECTION_FEEDBACK = "self_reflection_feedback";
    private static final String CTX_REFLECTION_APPLIED = "self_reflection_applied";

    private static final String REFLECTION_PROMPT = """
            你是一个质量审查员。请评估以下AI助手的回复质量。
            
            用户问题：%s
            AI回复：%s
            
            请从以下维度评分（每项1-10分）：
            1. **完整性**：是否完整回答了用户的问题
            2. **准确性**：信息是否有依据，是否存在编造
            3. **有用性**：回复对用户是否真正有帮助
            4. **安全性**：是否符合安全规范，无敏感信息泄露
            
            输出格式（严格遵循）：
            完整性=分数
            准确性=分数
            有用性=分数
            安全性=分数
            总评=分数
            改进建议=一句话建议（如果总评>=8则写"无需改进"）
            """;

    private static final String IMPROVEMENT_INSTRUCTION = """
            【自我反思改进指令】
            你的上一次回复经过质量审查，发现以下问题需要改进：
            %s
            
            请在回复中注意改进以上问题，确保：
            - 回答完整，不遗漏用户的关键问题
            - 所有信息都有明确来源（工具查询结果），不要编造
            - 给出具体可操作的建议，而非笼统的描述
            """;

    private SelfReflectionAdvisor(int order, ChatModel reflectionModel, int qualityThreshold) {
        this.order = order;
        this.reflectionModel = reflectionModel;
        this.qualityThreshold = qualityThreshold;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        Map<String, Object> context = new HashMap<>(request.context());

        // 检查是否有上一轮的反思反馈需要注入
        String feedback = (String) context.get(CTX_REFLECTION_FEEDBACK);
        Boolean applied = (Boolean) context.getOrDefault(CTX_REFLECTION_APPLIED, false);

        if (feedback != null && !applied) {
            log.info("注入自我反思改进指令");
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.add(new SystemMessage(String.format(IMPROVEMENT_INSTRUCTION, feedback)));
            context.put(CTX_REFLECTION_APPLIED, true);

            return ChatClientRequest.builder()
                    .prompt(request.prompt().withInstructions(messages))
                    .context(context)
                    .build();
        }

        return request;
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        if (reflectionModel == null) {
            return response;
        }

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
        String userInput = (String) context.getOrDefault("observability_user_input", "");

        // 异步执行反思评估
        evaluateAsync(context, userInput, output);

        return ChatClientResponse.builder()
                .chatResponse(chatResponse)
                .context(context)
                .build();
    }

    private void evaluateAsync(Map<String, Object> context, String userInput, String output) {
        Thread.startVirtualThread(() -> {
            try {
                ReflectionResult result = evaluate(userInput, output);
                context.put(CTX_REFLECTION_SCORE, result.totalScore);

                if (result.totalScore < qualityThreshold) {
                    log.warn("自我反思 - 质量评分低于阈值: {}/{}, 建议: {}",
                            result.totalScore, qualityThreshold, result.suggestion);
                    context.put(CTX_REFLECTION_FEEDBACK, result.suggestion);
                    context.put(CTX_REFLECTION_APPLIED, false);
                } else {
                    log.info("自我反思 - 质量评分通过: {}/{} [完整性={}, 准确性={}, 有用性={}, 安全性={}]",
                            result.totalScore, qualityThreshold,
                            result.completeness, result.accuracy,
                            result.helpfulness, result.safety);
                    context.remove(CTX_REFLECTION_FEEDBACK);
                }
            } catch (Exception e) {
                log.debug("自我反思评估跳过: {}", e.getMessage());
            }
        });
    }

    private ReflectionResult evaluate(String userInput, String output) {
        String prompt = String.format(REFLECTION_PROMPT,
                truncate(userInput, 300), truncate(output, 800));
        var llmResponse = reflectionModel.call(new Prompt(prompt));
        String result = llmResponse.getResult().getOutput().getText();

        return parseReflectionResult(result);
    }

    private ReflectionResult parseReflectionResult(String text) {
        ReflectionResult result = new ReflectionResult();
        if (text == null) return result;

        for (String line : text.split("\n")) {
            line = line.trim();
            try {
                if (line.startsWith("完整性=")) {
                    result.completeness = parseScore(line.substring(4));
                } else if (line.startsWith("准确性=")) {
                    result.accuracy = parseScore(line.substring(4));
                } else if (line.startsWith("有用性=")) {
                    result.helpfulness = parseScore(line.substring(4));
                } else if (line.startsWith("安全性=")) {
                    result.safety = parseScore(line.substring(4));
                } else if (line.startsWith("总评=")) {
                    result.totalScore = parseScore(line.substring(3));
                } else if (line.startsWith("改进建议=")) {
                    result.suggestion = line.substring(5);
                }
            } catch (Exception e) {
                // 解析单行失败，跳过
            }
        }

        // 如果总评未解析到，取四项平均
        if (result.totalScore == 0) {
            result.totalScore = (result.completeness + result.accuracy + 
                                  result.helpfulness + result.safety) / 4;
        }

        return result;
    }

    private int parseScore(String str) {
        str = str.trim().replaceAll("[^0-9]", "");
        return str.isEmpty() ? 5 : Math.min(10, Integer.parseInt(str));
    }

    private String truncate(String str, int maxLen) {
        if (str == null) return "";
        return str.length() > maxLen ? str.substring(0, maxLen) + "..." : str;
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder(ChatModel reflectionModel) {
        return new Builder(reflectionModel);
    }

    public static final class Builder {
        private int order = Ordered.LOWEST_PRECEDENCE - 20;
        private final ChatModel reflectionModel;
        private int qualityThreshold = 6;

        private Builder(ChatModel reflectionModel) {
            this.reflectionModel = reflectionModel;
        }

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder qualityThreshold(int qualityThreshold) {
            this.qualityThreshold = qualityThreshold;
            return this;
        }

        public SelfReflectionAdvisor build() {
            return new SelfReflectionAdvisor(order, reflectionModel, qualityThreshold);
        }
    }

    private static class ReflectionResult {
        int completeness = 5;
        int accuracy = 5;
        int helpfulness = 5;
        int safety = 5;
        int totalScore = 0;
        String suggestion = "无需改进";
    }
}
