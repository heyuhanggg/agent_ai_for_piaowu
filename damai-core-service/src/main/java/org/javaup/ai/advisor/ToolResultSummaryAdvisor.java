package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 工具结果摘要 Advisor
 * 
 * 当工具调用返回大量数据时（如日志查询、链路追踪），自动压缩工具结果，
 * 避免上下文窗口被低价值的原始数据占满。
 * 
 * 策略：
 * 1. 检测历史中的TOOL消息长度
 * 2. 超过阈值的工具结果使用LLM进行摘要
 * 3. 用摘要替代原始工具结果，保留关键信息
 * 4. 降级方案：截断 + 关键行提取
 */
@Slf4j
public class ToolResultSummaryAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final int toolResultMaxLength;
    private final ChatModel summaryModel;

    private static final String SUMMARY_PROMPT = """
            请将以下工具调用返回的原始数据压缩为简洁摘要。要求：
            1. 保留关键信息（错误信息、异常堆栈关键行、关键指标值、重要时间戳）
            2. 去除重复和冗余数据
            3. 保持数据的可追溯性（保留ID、traceId等标识）
            4. 摘要控制在200字以内
            
            === 工具返回原始数据 ===
            %s
            """;

    private ToolResultSummaryAdvisor(int order, int toolResultMaxLength, ChatModel summaryModel) {
        this.order = order;
        this.toolResultMaxLength = toolResultMaxLength;
        this.summaryModel = summaryModel;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        List<Message> messages = request.prompt().getInstructions();
        boolean needsCompression = false;

        // 检测是否有超长的TOOL消息
        for (Message msg : messages) {
            if (msg.getMessageType() == MessageType.TOOL && 
                msg.getText() != null && msg.getText().length() > toolResultMaxLength) {
                needsCompression = true;
                break;
            }
        }

        if (!needsCompression) {
            return request;
        }

        // 压缩超长的工具结果
        List<Message> compressedMessages = new ArrayList<>();
        int compressedCount = 0;

        for (Message msg : messages) {
            if (msg.getMessageType() == MessageType.TOOL && 
                msg.getText() != null && msg.getText().length() > toolResultMaxLength) {
                
                String summary = summarizeToolResult(msg.getText());
                compressedMessages.add(new SystemMessage(
                        "[工具结果摘要] " + summary));
                compressedCount++;
            } else {
                compressedMessages.add(msg);
            }
        }

        if (compressedCount > 0) {
            log.info("压缩了 {} 个超长工具结果 (阈值: {} 字符)", compressedCount, toolResultMaxLength);
        }

        return ChatClientRequest.builder()
                .prompt(request.prompt().withInstructions(compressedMessages))
                .context(request.context())
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        return response;
    }

    private String summarizeToolResult(String toolResult) {
        try {
            String prompt = String.format(SUMMARY_PROMPT, 
                    toolResult.length() > 3000 ? toolResult.substring(0, 3000) + "\n...[截断]" : toolResult);
            var response = summaryModel.call(new Prompt(prompt));
            return response.getResult().getOutput().getText();
        } catch (Exception e) {
            log.warn("工具结果LLM摘要失败，使用降级方案: {}", e.getMessage());
            return fallbackSummary(toolResult);
        }
    }

    private String fallbackSummary(String toolResult) {
        StringBuilder sb = new StringBuilder();
        String[] lines = toolResult.split("\n");

        // 提取含错误/异常/关键指标的行
        int kept = 0;
        for (String line : lines) {
            String lower = line.toLowerCase();
            if (lower.contains("error") || lower.contains("exception") || lower.contains("fail") ||
                lower.contains("timeout") || lower.contains("traceId") || lower.contains("warn") ||
                lower.contains("异常") || lower.contains("错误") || lower.contains("超时")) {
                sb.append(line.length() > 200 ? line.substring(0, 200) + "..." : line).append("\n");
                kept++;
                if (kept >= 15) break;
            }
        }

        if (kept == 0) {
            // 没有关键行，取前5行和后2行
            for (int i = 0; i < Math.min(5, lines.length); i++) {
                sb.append(lines[i]).append("\n");
            }
            if (lines.length > 7) {
                sb.append("...(共").append(lines.length).append("行)...\n");
                for (int i = Math.max(5, lines.length - 2); i < lines.length; i++) {
                    sb.append(lines[i]).append("\n");
                }
            }
        }

        return sb.toString().trim();
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder(ChatModel summaryModel) {
        return new Builder(summaryModel);
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 55;
        private int toolResultMaxLength = 1500;
        private final ChatModel summaryModel;

        private Builder(ChatModel summaryModel) {
            this.summaryModel = summaryModel;
        }

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder toolResultMaxLength(int toolResultMaxLength) {
            this.toolResultMaxLength = toolResultMaxLength;
            return this;
        }

        public ToolResultSummaryAdvisor build() {
            return new ToolResultSummaryAdvisor(order, toolResultMaxLength, summaryModel);
        }
    }
}
