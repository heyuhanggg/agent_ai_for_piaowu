package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 动态上下文压缩 Advisor
 * 
 * 当对话历史超过阈值时，自动使用LLM对较早的消息进行摘要压缩，
 * 保留最近的消息不变，从而在有限的上下文窗口内保留尽可能多的有效信息。
 * 
 * 压缩策略：
 * 1. 保留最近 N 条消息不压缩（preserveRecentCount）
 * 2. 当历史消息总数超过阈值时触发压缩（compressionThreshold）
 * 3. 使用LLM将早期消息压缩为一段结构化摘要
 * 4. 摘要作为SystemMessage注入到对话开头，替代原始的早期消息
 */
@Slf4j
public class ContextCompressionAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final int compressionThreshold;
    private final int preserveRecentCount;
    private final ChatModel compressionModel;

    private static final String CTX_COMPRESSION_COUNT = "context_compression_count";
    private static final String CTX_COMPRESSED_SUMMARY = "context_compressed_summary";

    private static final String COMPRESSION_PROMPT = """
            请将以下对话历史压缩为一段简洁的结构化摘要。要求：
            1. 保留所有关键信息：用户意图、已确认的参数、工具调用结果、决策结论
            2. 使用结构化格式，便于后续对话参考
            3. 丢弃闲聊、重复确认等低信息量内容
            4. 摘要长度控制在300字以内
            
            输出格式：
            【用户意图】...
            【已收集信息】...
            【已执行操作】...
            【当前状态】...
            
            === 对话历史 ===
            %s
            """;

    private ContextCompressionAdvisor(int order, int compressionThreshold,
                                      int preserveRecentCount, ChatModel compressionModel) {
        this.order = order;
        this.compressionThreshold = compressionThreshold;
        this.preserveRecentCount = preserveRecentCount;
        this.compressionModel = compressionModel;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        List<Message> messages = new ArrayList<>(request.prompt().getInstructions());

        // 分离系统消息和对话消息
        List<Message> systemMessages = new ArrayList<>();
        List<Message> conversationMessages = new ArrayList<>();
        for (Message msg : messages) {
            if (msg.getMessageType() == MessageType.SYSTEM) {
                systemMessages.add(msg);
            } else {
                conversationMessages.add(msg);
            }
        }

        // 未达到压缩阈值，直接通过
        if (conversationMessages.size() <= compressionThreshold) {
            return request;
        }

        log.info("对话历史 {} 条超过阈值 {}，触发上下文压缩", 
                conversationMessages.size(), compressionThreshold);

        // 分割：早期消息 vs 保留的近期消息
        int splitPoint = conversationMessages.size() - preserveRecentCount;
        List<Message> earlyMessages = conversationMessages.subList(0, splitPoint);
        List<Message> recentMessages = conversationMessages.subList(splitPoint, conversationMessages.size());

        // 压缩早期消息
        String summary = compressMessages(earlyMessages);

        // 重建消息列表：系统消息 + 压缩摘要 + 近期消息
        List<Message> compressedMessages = new ArrayList<>(systemMessages);
        compressedMessages.add(new SystemMessage(
                "【历史对话摘要】以下是之前对话的压缩摘要，请参考：\n" + summary));
        compressedMessages.addAll(recentMessages);

        Map<String, Object> context = new HashMap<>(request.context());
        int compressionCount = (int) context.getOrDefault(CTX_COMPRESSION_COUNT, 0);
        context.put(CTX_COMPRESSION_COUNT, compressionCount + 1);
        context.put(CTX_COMPRESSED_SUMMARY, summary);

        log.info("上下文压缩完成: {} 条消息压缩为摘要, 保留 {} 条近期消息 (第{}次压缩)",
                earlyMessages.size(), recentMessages.size(), compressionCount + 1);

        return ChatClientRequest.builder()
                .prompt(request.prompt().withInstructions(compressedMessages))
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        return response;
    }

    private String compressMessages(List<Message> messages) {
        String conversationText = messages.stream()
                .map(msg -> {
                    String role = switch (msg.getMessageType()) {
                        case USER -> "用户";
                        case ASSISTANT -> "助手";
                        case TOOL -> "工具结果";
                        default -> "系统";
                    };
                    return role + ": " + msg.getText();
                })
                .collect(Collectors.joining("\n"));

        try {
            String prompt = String.format(COMPRESSION_PROMPT, conversationText);
            var response = compressionModel.call(new Prompt(prompt));
            String summary = response.getResult().getOutput().getText();
            log.debug("压缩摘要: {}", summary);
            return summary;
        } catch (Exception e) {
            log.warn("LLM压缩失败，使用降级方案: {}", e.getMessage());
            return fallbackCompression(messages);
        }
    }

    private String fallbackCompression(List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        sb.append("【历史摘要-降级模式】\n");
        for (Message msg : messages) {
            if (msg.getMessageType() == MessageType.USER) {
                String text = msg.getText();
                sb.append("- 用户: ").append(text.length() > 80 ? text.substring(0, 80) + "..." : text).append("\n");
            } else if (msg.getMessageType() == MessageType.ASSISTANT) {
                String text = msg.getText();
                sb.append("- 助手: ").append(text.length() > 80 ? text.substring(0, 80) + "..." : text).append("\n");
            }
        }
        return sb.toString();
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder(ChatModel compressionModel) {
        return new Builder(compressionModel);
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 50;
        private int compressionThreshold = 12;
        private int preserveRecentCount = 4;
        private final ChatModel compressionModel;

        private Builder(ChatModel compressionModel) {
            this.compressionModel = compressionModel;
        }

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder compressionThreshold(int compressionThreshold) {
            this.compressionThreshold = compressionThreshold;
            return this;
        }

        public Builder preserveRecentCount(int preserveRecentCount) {
            this.preserveRecentCount = preserveRecentCount;
            return this;
        }

        public ContextCompressionAdvisor build() {
            return new ContextCompressionAdvisor(order, compressionThreshold, 
                    preserveRecentCount, compressionModel);
        }
    }
}
