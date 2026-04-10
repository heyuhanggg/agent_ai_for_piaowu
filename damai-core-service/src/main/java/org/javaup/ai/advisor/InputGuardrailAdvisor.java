package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * 输入护栏 Advisor
 * 
 * 在用户输入到达模型之前进行多层安全检查：
 * 1. 规则层：正则匹配敏感关键词、SQL注入、Prompt注入等
 * 2. 意图分类：判断用户输入是否在允许的话题范围内
 * 3. LLM层：使用轻量级LLM判断复杂的边界情况
 * 
 * 如果检测到危险输入，直接返回安全回复，不传递给下游模型。
 */
@Slf4j
public class InputGuardrailAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final ChatModel guardrailModel;
    private final boolean enableLlmCheck;
    private final List<String> allowedTopics;

    private static final String CTX_GUARDRAIL_TRIGGERED = "input_guardrail_triggered";
    private static final String CTX_GUARDRAIL_REASON = "input_guardrail_reason";

    // 常见的Prompt注入模式
    private static final List<Pattern> INJECTION_PATTERNS = List.of(
            Pattern.compile("(?i)ignore\\s+(previous|above|all)\\s+(instructions?|prompts?|rules?)"),
            Pattern.compile("(?i)forget\\s+(your|all|previous)\\s+(instructions?|rules?|prompts?)"),
            Pattern.compile("(?i)you\\s+are\\s+now\\s+a"),
            Pattern.compile("(?i)new\\s+instructions?\\s*:"),
            Pattern.compile("(?i)system\\s*:\\s*you\\s+are"),
            Pattern.compile("(?i)disregard\\s+(all|any|previous)"),
            Pattern.compile("忽略(之前|上面|所有)(的)?(指令|规则|提示|设定)"),
            Pattern.compile("你现在(是|扮演|变成)"),
            Pattern.compile("新(的)?指令\\s*[：:]"),
            Pattern.compile("从现在开始你(是|要)")
    );

    // 敏感信息模式
    private static final List<Pattern> SENSITIVE_PATTERNS = List.of(
            Pattern.compile("(?i)(api[_\\s-]?key|secret|password|token)\\s*[=:]\\s*\\S+"),
            Pattern.compile("(?i)(drop|delete|truncate|alter)\\s+table"),
            Pattern.compile("(?i)\\b(exec|execute|eval)\\s*\\(")
    );

    private static final String INTENT_CHECK_PROMPT = """
            判断以下用户输入是否属于合法的业务请求。
            
            允许的话题范围：%s
            
            用户输入：%s
            
            请回答：
            - SAFE：属于合法业务请求
            - UNSAFE：不属于合法范围，并简述原因
            
            只输出 SAFE 或 UNSAFE:原因
            """;

    private InputGuardrailAdvisor(int order, ChatModel guardrailModel,
                                   boolean enableLlmCheck, List<String> allowedTopics) {
        this.order = order;
        this.guardrailModel = guardrailModel;
        this.enableLlmCheck = enableLlmCheck;
        this.allowedTopics = allowedTopics;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        String userInput = request.prompt().getUserMessage().getText();
        Map<String, Object> context = new HashMap<>(request.context());

        // 第一层：规则检查 - Prompt注入
        for (Pattern pattern : INJECTION_PATTERNS) {
            if (pattern.matcher(userInput).find()) {
                log.warn("输入护栏触发 - Prompt注入检测: 匹配模式 {}", pattern.pattern());
                return buildBlockedRequest(request, context,
                        "检测到不安全的输入模式，请正常提问哦~");
            }
        }

        // 第二层：规则检查 - 敏感信息
        for (Pattern pattern : SENSITIVE_PATTERNS) {
            if (pattern.matcher(userInput).find()) {
                log.warn("输入护栏触发 - 敏感信息检测: 匹配模式 {}", pattern.pattern());
                return buildBlockedRequest(request, context,
                        "检测到可能包含敏感信息的输入，请不要在对话中发送密码、密钥等信息。");
            }
        }

        // 第三层：输入长度检查
        if (userInput.length() > 5000) {
            log.warn("输入护栏触发 - 输入过长: {} 字符", userInput.length());
            return buildBlockedRequest(request, context,
                    "输入内容过长，请精简您的问题后重新提问。");
        }

        // 第四层：LLM意图分类（可选）
        if (enableLlmCheck && guardrailModel != null && !allowedTopics.isEmpty()) {
            String checkResult = llmIntentCheck(userInput);
            if (checkResult != null && checkResult.startsWith("UNSAFE")) {
                String reason = checkResult.contains(":") ? checkResult.split(":", 2)[1].trim() : "超出服务范围";
                log.warn("输入护栏触发 - LLM意图检测: {}", reason);
                return buildBlockedRequest(request, context,
                        "抱歉，您的问题超出了我的服务范围。我可以帮您处理" +
                        String.join("、", allowedTopics) + "相关的问题。");
            }
        }

        // 所有检查通过
        context.put(CTX_GUARDRAIL_TRIGGERED, false);
        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        return response;
    }

    private ChatClientRequest buildBlockedRequest(ChatClientRequest originalRequest,
                                                   Map<String, Object> context, String safeResponse) {
        context.put(CTX_GUARDRAIL_TRIGGERED, true);
        context.put(CTX_GUARDRAIL_REASON, safeResponse);

        // 注入安全回复指令，让模型直接输出安全回复
        List<Message> messages = new ArrayList<>();
        messages.add(new SystemMessage(
                "用户的输入触发了安全检查，请直接回复以下内容，不要做其他操作：\n" + safeResponse));

        return ChatClientRequest.builder()
                .prompt(originalRequest.prompt().withInstructions(messages))
                .context(context)
                .build();
    }

    private String llmIntentCheck(String userInput) {
        try {
            String topicList = String.join("、", allowedTopics);
            String prompt = String.format(INTENT_CHECK_PROMPT, topicList,
                    userInput.length() > 300 ? userInput.substring(0, 300) : userInput);
            var response = guardrailModel.call(new Prompt(prompt));
            return response.getResult().getOutput().getText().trim();
        } catch (Exception e) {
            log.debug("LLM意图检查跳过: {}", e.getMessage());
            return "SAFE";
        }
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 10;
        private ChatModel guardrailModel;
        private boolean enableLlmCheck = false;
        private List<String> allowedTopics = List.of("节目查询", "购票", "订单", "日志查询", "运维分析");

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder guardrailModel(ChatModel guardrailModel) {
            this.guardrailModel = guardrailModel;
            return this;
        }

        public Builder enableLlmCheck(boolean enableLlmCheck) {
            this.enableLlmCheck = enableLlmCheck;
            return this;
        }

        public Builder allowedTopics(List<String> allowedTopics) {
            this.allowedTopics = allowedTopics;
            return this;
        }

        public InputGuardrailAdvisor build() {
            return new InputGuardrailAdvisor(order, guardrailModel, enableLlmCheck, allowedTopics);
        }
    }
}
