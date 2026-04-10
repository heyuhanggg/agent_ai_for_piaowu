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
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 对话消息重要度评分 + 加权保留 Advisor
 *
 * 上下文工程的核心改进——不再粗暴地按时间截断消息，而是给每条消息打重要度分数，
 * 当上下文接近容量上限时，优先保留高分消息，淘汰低分消息。
 *
 * 评分维度（规则打分，不调LLM，零额外开销）：
 * 1. **信息密度**：包含实体（手机号/traceId/服务名/城市）的消息 +3分
 * 2. **决策关键性**：包含工具调用结果的消息 +4分（诊断证据不可丢失）
 * 3. **用户意图锚点**：用户的首条消息和最近确认消息 +5分
 * 4. **时间衰减**：越早的消息分数衰减越多（每隔5轮 -1分）
 * 5. **闲聊惩罚**：纯寒暄/礼貌用语的消息 -3分
 *
 * 当消息数超过 maxMessages 时，按分数排序，保留 Top-N，丢弃的消息聚合成一行摘要注入。
 */
@Slf4j
public class ConversationImportanceAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final int maxMessages;
    private final int protectedRecentCount;

    private static final String CTX_IMPORTANCE_APPLIED = "conversation_importance_applied";

    private ConversationImportanceAdvisor(int order, int maxMessages, int protectedRecentCount) {
        this.order = order;
        this.maxMessages = maxMessages;
        this.protectedRecentCount = protectedRecentCount;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        List<Message> messages = new ArrayList<>(request.prompt().getInstructions());

        // 分离 SystemMessage 和对话消息
        List<Message> systemMessages = new ArrayList<>();
        List<Message> conversationMessages = new ArrayList<>();
        for (Message msg : messages) {
            if (msg instanceof SystemMessage) {
                systemMessages.add(msg);
            } else {
                conversationMessages.add(msg);
            }
        }

        // 未超限不处理
        if (conversationMessages.size() <= maxMessages) {
            return request;
        }

        Map<String, Object> context = new HashMap<>(request.context());
        if (context.containsKey(CTX_IMPORTANCE_APPLIED)) {
            return request;
        }

        int totalSize = conversationMessages.size();

        // 保护最近N条消息（绝对不淘汰）
        List<Message> protectedMessages = new ArrayList<>();
        List<Message> candidateMessages = new ArrayList<>();
        for (int i = 0; i < totalSize; i++) {
            if (i >= totalSize - protectedRecentCount) {
                protectedMessages.add(conversationMessages.get(i));
            } else {
                candidateMessages.add(conversationMessages.get(i));
            }
        }

        // 对候选消息打分
        List<ScoredMessage> scoredMessages = new ArrayList<>();
        for (int i = 0; i < candidateMessages.size(); i++) {
            Message msg = candidateMessages.get(i);
            int score = scoreMessage(msg, i, candidateMessages.size());
            scoredMessages.add(new ScoredMessage(msg, score, i));
        }

        // 按分数降序排序，保留 Top (maxMessages - protectedRecentCount - systemMessages.size())
        int keepCount = Math.max(1, maxMessages - protectedRecentCount - systemMessages.size());
        scoredMessages.sort(Comparator.comparingInt(ScoredMessage::score).reversed());

        List<ScoredMessage> retained = scoredMessages.subList(0, Math.min(keepCount, scoredMessages.size()));
        List<ScoredMessage> dropped = scoredMessages.size() > keepCount ?
                scoredMessages.subList(keepCount, scoredMessages.size()) : List.of();

        // 按原始顺序恢复被保留的消息
        retained.sort(Comparator.comparingInt(ScoredMessage::originalIndex));
        List<Message> finalMessages = new ArrayList<>(systemMessages);

        // 对丢弃的消息生成一行摘要
        if (!dropped.isEmpty()) {
            String droppedSummary = buildDroppedSummary(dropped);
            finalMessages.add(new SystemMessage(
                    String.format("[上下文摘要] 以下是早期对话中的关键信息（%d条低优先级消息已压缩）：%s",
                            dropped.size(), droppedSummary)));
            log.info("ConversationImportance: 保留 {} 条高分消息，压缩 {} 条低分消息",
                    retained.size(), dropped.size());
        }

        for (ScoredMessage sm : retained) {
            finalMessages.add(sm.message());
        }
        finalMessages.addAll(protectedMessages);

        context.put(CTX_IMPORTANCE_APPLIED, true);
        return ChatClientRequest.builder()
                .prompt(request.prompt().withInstructions(finalMessages))
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        return response;
    }

    // ─────────── 评分逻辑（纯规则，零LLM开销） ───────────

    private int scoreMessage(Message msg, int positionIndex, int totalCount) {
        int score = 5; // 基础分
        String text = getMessageText(msg);
        if (text == null || text.isEmpty()) return 1;

        // 1. 信息密度：包含关键实体 +3
        if (containsEntity(text)) {
            score += 3;
        }

        // 2. 决策关键性：工具调用结果 +4
        if (msg instanceof AssistantMessage am) {
            if (am.getToolCalls() != null && !am.getToolCalls().isEmpty()) {
                score += 4;
            }
            // 包含结构化分析结果
            if (text.contains("根因") || text.contains("建议") || text.contains("结论")) {
                score += 2;
            }
        }

        // 3. 用户意图锚点：首条用户消息 +5
        if (msg instanceof UserMessage && positionIndex == 0) {
            score += 5;
        }

        // 用户确认/关键决策消息 +3
        if (msg instanceof UserMessage) {
            String lower = text.toLowerCase();
            if (lower.contains("确认") || lower.contains("没错") || lower.contains("是的")
                    || lower.contains("对") || lower.contains("好的") || lower.contains("就这个")) {
                score += 3;
            }
            // 包含关键业务信息的用户消息
            if (lower.contains("手机") || lower.contains("证件") || lower.contains("买")
                    || lower.contains("订单") || lower.contains("traceId") || lower.contains("traceid")) {
                score += 3;
            }
        }

        // 4. 时间衰减：每隔5条 -1分
        int decay = (totalCount - positionIndex) / 5;
        score -= decay;

        // 5. 闲聊惩罚
        if (isSmallTalk(text)) {
            score -= 3;
        }

        return Math.max(1, score);
    }

    private boolean containsEntity(String text) {
        // 手机号
        if (text.matches(".*1[3-9]\\d{9}.*")) return true;
        // 身份证号
        if (text.matches(".*\\d{17}[\\dXx].*")) return true;
        // traceId（32位十六进制）
        if (text.matches(".*[a-fA-F0-9]{32}.*")) return true;
        // 服务名格式
        if (text.matches(".*[a-z]+-service.*")) return true;
        // 订单号
        if (text.matches(".*\\d{16,20}.*")) return true;
        return false;
    }

    private boolean isSmallTalk(String text) {
        if (text.length() > 30) return false;
        String lower = text.toLowerCase().trim();
        String[] smallTalkPatterns = {
                "你好", "hello", "hi", "嗨", "谢谢", "感谢", "好的", "ok", "明白",
                "知道了", "了解", "嗯", "哦", "再见", "拜拜", "bye"
        };
        for (String pattern : smallTalkPatterns) {
            if (lower.equals(pattern) || lower.startsWith(pattern + "，") || lower.startsWith(pattern + "。")) {
                return true;
            }
        }
        return false;
    }

    private String getMessageText(Message msg) {
        if (msg instanceof UserMessage um) return um.getText();
        if (msg instanceof AssistantMessage am) return am.getText();
        if (msg instanceof SystemMessage sm) return sm.getText();
        return "";
    }

    private String buildDroppedSummary(List<ScoredMessage> dropped) {
        StringBuilder sb = new StringBuilder();
        for (ScoredMessage sm : dropped) {
            String text = getMessageText(sm.message());
            if (text != null && !text.isEmpty() && !isSmallTalk(text)) {
                String snippet = text.length() > 40 ? text.substring(0, 40) + "..." : text;
                sb.append(" [").append(snippet).append("]");
            }
        }
        return sb.isEmpty() ? " (均为寒暄/闲聊)" : sb.toString();
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 60;
        private int maxMessages = 20;
        private int protectedRecentCount = 4;

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder maxMessages(int maxMessages) {
            this.maxMessages = maxMessages;
            return this;
        }

        public Builder protectedRecentCount(int count) {
            this.protectedRecentCount = count;
            return this;
        }

        public ConversationImportanceAdvisor build() {
            return new ConversationImportanceAdvisor(order, maxMessages, protectedRecentCount);
        }
    }

    private record ScoredMessage(Message message, int score, int originalIndex) {}
}
