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
import java.util.concurrent.ConcurrentHashMap;

/**
 * 实体记忆 Advisor
 * 
 * 从对话中自动提取关键实体（用户信息、节目偏好、操作状态等），
 * 维护一个结构化的实体记忆表，并在每轮对话中注入到上下文中。
 * 
 * 功能：
 * 1. 从用户输入和助手回复中提取关键实体
 * 2. 维护跨轮次的实体记忆（城市、节目、票档、手机号等）
 * 3. 将实体记忆作为SystemMessage注入，减少重复询问
 * 4. 支持实体过期和更新
 */
@Slf4j
public class EntityMemoryAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final ChatModel extractionModel;
    private final ConcurrentHashMap<String, Map<String, String>> conversationEntities;

    private static final String CTX_ENTITY_MEMORY = "entity_memory";

    private static final String EXTRACTION_PROMPT = """
            从以下对话中提取关键实体信息。只输出有明确值的实体，格式为 key=value，每行一个。
            
            可提取的实体类型：
            - city: 城市名
            - program_name: 节目/演唱会名称
            - artist: 艺人/明星名
            - program_type: 节目类型(演唱会/话剧/音乐剧等)
            - ticket_category: 票档信息
            - phone: 手机号
            - id_card: 证件号
            - ticket_count: 购票数量
            - date: 日期/时间
            - service_name: 微服务名称
            - log_level: 日志级别
            - trace_id: 链路追踪ID
            - error_keyword: 错误关键词
            
            如果没有新实体，输出 NONE
            
            === 对话内容 ===
            用户: %s
            助手: %s
            """;

    private EntityMemoryAdvisor(int order, ChatModel extractionModel) {
        this.order = order;
        this.extractionModel = extractionModel;
        this.conversationEntities = new ConcurrentHashMap<>();
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        Map<String, Object> context = new HashMap<>(request.context());
        String conversationId = getConversationId(context, "default");

        // 获取当前会话的实体记忆
        Map<String, String> entities = conversationEntities.getOrDefault(conversationId, new HashMap<>());

        if (!entities.isEmpty()) {
            // 将实体记忆注入为SystemMessage
            String entityContext = buildEntityContext(entities);
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.add(new SystemMessage(entityContext));

            log.debug("注入实体记忆: {} 个实体 -> 会话 {}", entities.size(), conversationId);

            context.put(CTX_ENTITY_MEMORY, new HashMap<>(entities));

            return ChatClientRequest.builder()
                    .prompt(request.prompt().withInstructions(messages))
                    .context(context)
                    .build();
        }

        return request;
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        Map<String, Object> context = response.context();
        String conversationId = getConversationId(context, "default");

        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }

        String assistantOutput = chatResponse.getResult().getOutput().getText();
        String userInput = (String) context.getOrDefault("observability_user_input", "");

        // 异步提取实体
        extractEntitiesAsync(conversationId, userInput, assistantOutput);

        return response;
    }

    private void extractEntitiesAsync(String conversationId, String userInput, String assistantOutput) {
        Thread.startVirtualThread(() -> {
            try {
                Map<String, String> newEntities = extractEntities(userInput, assistantOutput);
                if (!newEntities.isEmpty()) {
                    Map<String, String> existing = conversationEntities.computeIfAbsent(
                            conversationId, k -> new ConcurrentHashMap<>());
                    existing.putAll(newEntities);
                    log.info("实体提取完成: 新增/更新 {} 个实体, 会话 {} 当前共 {} 个",
                            newEntities.size(), conversationId, existing.size());
                }
            } catch (Exception e) {
                log.warn("实体提取失败: {}", e.getMessage());
            }
        });
    }

    private Map<String, String> extractEntities(String userInput, String assistantOutput) {
        Map<String, String> entities = new HashMap<>();

        // 先用规则快速提取常见模式
        entities.putAll(ruleBasedExtraction(userInput));

        // 再用LLM提取复杂实体
        try {
            String prompt = String.format(EXTRACTION_PROMPT,
                    truncate(userInput, 500), truncate(assistantOutput, 500));
            var response = extractionModel.call(new Prompt(prompt));
            String result = response.getResult().getOutput().getText();

            if (result != null && !result.trim().equals("NONE")) {
                for (String line : result.split("\n")) {
                    line = line.trim();
                    if (line.contains("=")) {
                        String[] parts = line.split("=", 2);
                        if (parts.length == 2 && !parts[1].trim().isEmpty()) {
                            entities.put(parts[0].trim(), parts[1].trim());
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.debug("LLM实体提取跳过: {}", e.getMessage());
        }

        return entities;
    }

    private Map<String, String> ruleBasedExtraction(String input) {
        Map<String, String> entities = new HashMap<>();
        if (input == null) return entities;

        // 手机号
        var phoneMatcher = java.util.regex.Pattern.compile("1[3-9]\\d{9}").matcher(input);
        if (phoneMatcher.find()) {
            entities.put("phone", phoneMatcher.group());
        }

        // 身份证号
        var idMatcher = java.util.regex.Pattern.compile("\\d{17}[\\dXx]").matcher(input);
        if (idMatcher.find()) {
            entities.put("id_card", idMatcher.group());
        }

        // traceId (常见格式)
        var traceMatcher = java.util.regex.Pattern.compile("[a-f0-9]{32}|[a-f0-9-]{36}").matcher(input);
        if (traceMatcher.find()) {
            entities.put("trace_id", traceMatcher.group());
        }

        return entities;
    }

    private String buildEntityContext(Map<String, String> entities) {
        StringBuilder sb = new StringBuilder("【已知实体记忆】当前会话已收集到以下信息，无需重复询问：\n");
        for (Map.Entry<String, String> entry : entities.entrySet()) {
            String label = switch (entry.getKey()) {
                case "city" -> "城市";
                case "program_name" -> "节目名称";
                case "artist" -> "艺人";
                case "program_type" -> "节目类型";
                case "ticket_category" -> "票档";
                case "phone" -> "手机号";
                case "id_card" -> "证件号";
                case "ticket_count" -> "购票数量";
                case "date" -> "日期";
                case "service_name" -> "服务名";
                case "log_level" -> "日志级别";
                case "trace_id" -> "链路ID";
                case "error_keyword" -> "错误关键词";
                default -> entry.getKey();
            };
            sb.append("- ").append(label).append(": ").append(entry.getValue()).append("\n");
        }
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

    public static Builder builder(ChatModel extractionModel) {
        return new Builder(extractionModel);
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 60;
        private final ChatModel extractionModel;

        private Builder(ChatModel extractionModel) {
            this.extractionModel = extractionModel;
        }

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public EntityMemoryAdvisor build() {
            return new EntityMemoryAdvisor(order, extractionModel);
        }
    }
}
