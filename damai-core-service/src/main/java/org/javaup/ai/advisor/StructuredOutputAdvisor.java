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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 结构化输出校验 + 自动修复 Advisor (Guardrails 工程)
 *
 * 确保模型输出符合预期格式，对不合格输出进行自动修复：
 *
 * 1. **Markdown 结构校验**：检查诊断报告是否包含必要章节（问题现象/根因分析/解决方案）
 * 2. **关键字段完整性校验**：购票确认是否包含手机号、票档、数量等必填字段
 * 3. **JSON 格式修复**：如果模型输出了残缺JSON（缺尾括号），尝试自动补全
 * 4. **格式降级**：如果LLM修复也失败，追加友好提示而非返回乱码
 * 5. **一致性保障**：对同一类查询保证输出风格统一（表格/树形图/列表）
 */
@Slf4j
public class StructuredOutputAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final ChatModel repairModel;

    private static final String CTX_OUTPUT_VALIDATED = "structured_output_validated";
    private static final String CTX_EXPECTED_FORMAT  = "structured_output_expected_format";

    // 诊断报告必须包含的章节关键词
    private static final List<String> DIAGNOSIS_REQUIRED_SECTIONS = List.of(
            "问题", "分析", "原因", "根因", "建议", "方案", "解决"
    );

    // 购票确认必须包含的字段
    private static final List<String> PURCHASE_REQUIRED_FIELDS = List.of(
            "节目", "时间", "票档", "数量", "手机"
    );

    private static final String REPAIR_PROMPT = """
            以下AI助手的回复格式不够规范。请帮忙修正格式问题，保持内容不变。
            
            格式问题：%s
            原始回复：%s
            
            请输出修正后的完整回复（只输出修正结果，不要解释）：
            """;

    private StructuredOutputAdvisor(int order, ChatModel repairModel) {
        this.order = order;
        this.repairModel = repairModel;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        // 根据用户输入预判需要的输出格式
        Map<String, Object> context = new HashMap<>(request.context());

        List<Message> messages = request.prompt().getInstructions();
        String userText = extractLastUserText(messages);
        if (userText != null) {
            OutputFormat format = detectExpectedFormat(userText);
            if (format != OutputFormat.NONE) {
                context.put(CTX_EXPECTED_FORMAT, format.name());
                log.debug("StructuredOutput: 预判输出格式={}", format);
            }
        }

        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }

        Map<String, Object> context = new HashMap<>(response.context());
        if (context.containsKey(CTX_OUTPUT_VALIDATED)) {
            return response;
        }

        String output = chatResponse.getResult().getOutput().getText();
        if (output == null || output.isEmpty()) {
            return response;
        }

        String expectedFormatStr = (String) context.getOrDefault(CTX_EXPECTED_FORMAT, "NONE");
        OutputFormat expectedFormat;
        try {
            expectedFormat = OutputFormat.valueOf(expectedFormatStr);
        } catch (IllegalArgumentException e) {
            expectedFormat = OutputFormat.NONE;
        }

        // 执行格式校验
        ValidationResult validation = validateOutput(output, expectedFormat);

        if (validation.isValid) {
            context.put(CTX_OUTPUT_VALIDATED, true);
            return ChatClientResponse.builder().chatResponse(chatResponse).context(context).build();
        }

        log.warn("StructuredOutput: 检测到格式问题: {}", validation.issue);

        // 尝试自动修复
        String repaired = attemptRepair(output, validation);
        if (repaired != null && !repaired.equals(output)) {
            log.info("StructuredOutput: 格式已自动修复");
            ChatResponse repairedResponse = buildRepairedResponse(chatResponse, repaired);
            context.put(CTX_OUTPUT_VALIDATED, true);
            return ChatClientResponse.builder().chatResponse(repairedResponse).context(context).build();
        }

        // 修复失败，追加格式提示（降级策略）
        String hint = buildFormatHint(validation, expectedFormat);
        if (hint != null) {
            String enhanced = output + "\n\n" + hint;
            ChatResponse hintResponse = buildRepairedResponse(chatResponse, enhanced);
            context.put(CTX_OUTPUT_VALIDATED, true);
            return ChatClientResponse.builder().chatResponse(hintResponse).context(context).build();
        }

        context.put(CTX_OUTPUT_VALIDATED, true);
        return ChatClientResponse.builder().chatResponse(chatResponse).context(context).build();
    }

    // ─────────── 格式校验逻辑 ───────────

    private ValidationResult validateOutput(String output, OutputFormat format) {
        // 1. 残缺JSON检测
        if (hasIncompleteJson(output)) {
            return new ValidationResult(false, "JSON_INCOMPLETE", "输出中包含未闭合的JSON结构");
        }

        // 2. 按预期格式校验
        switch (format) {
            case DIAGNOSIS_REPORT:
                int matchedSections = 0;
                for (String keyword : DIAGNOSIS_REQUIRED_SECTIONS) {
                    if (output.contains(keyword)) matchedSections++;
                }
                if (matchedSections < 3) {
                    return new ValidationResult(false, "INCOMPLETE_DIAGNOSIS",
                            String.format("诊断报告缺少关键章节（仅命中%d/%d个关键词）",
                                    matchedSections, DIAGNOSIS_REQUIRED_SECTIONS.size()));
                }
                break;

            case PURCHASE_CONFIRM:
                List<String> missingFields = new ArrayList<>();
                for (String field : PURCHASE_REQUIRED_FIELDS) {
                    if (!output.contains(field)) {
                        missingFields.add(field);
                    }
                }
                if (!missingFields.isEmpty()) {
                    return new ValidationResult(false, "MISSING_PURCHASE_FIELDS",
                            "购票确认缺少字段: " + String.join(", ", missingFields));
                }
                break;

            case TABLE:
                if (!output.contains("|") || !output.contains("---")) {
                    return new ValidationResult(false, "MISSING_TABLE",
                            "预期表格输出但未检测到Markdown表格格式");
                }
                break;

            default:
                break;
        }

        return new ValidationResult(true, null, null);
    }

    private boolean hasIncompleteJson(String output) {
        int braces = 0, brackets = 0;
        boolean inJson = false;
        for (char c : output.toCharArray()) {
            if (c == '{') { braces++; inJson = true; }
            if (c == '}') braces--;
            if (c == '[') { brackets++; inJson = true; }
            if (c == ']') brackets--;
        }
        return inJson && (braces != 0 || brackets != 0);
    }

    // ─────────── 自动修复 ───────────

    private String attemptRepair(String output, ValidationResult validation) {
        // 1. JSON 不完整 → 尝试补全括号
        if ("JSON_INCOMPLETE".equals(validation.issueType)) {
            return tryFixIncompleteJson(output);
        }

        // 2. 用 LLM 修复格式（如果配置了修复模型）
        if (repairModel != null) {
            try {
                Prompt repairPrompt = new Prompt(
                        String.format(REPAIR_PROMPT, validation.issue, truncate(output, 2000)));
                ChatResponse repairResponse = repairModel.call(repairPrompt);
                if (repairResponse != null && repairResponse.getResult() != null) {
                    String repaired = repairResponse.getResult().getOutput().getText();
                    if (repaired != null && !repaired.isBlank()) {
                        return repaired;
                    }
                }
            } catch (Exception e) {
                log.warn("StructuredOutput: LLM修复失败，降级放行: {}", e.getMessage());
            }
        }

        return null;
    }

    private String tryFixIncompleteJson(String output) {
        StringBuilder sb = new StringBuilder(output);
        int braces = 0, brackets = 0;
        for (char c : output.toCharArray()) {
            if (c == '{') braces++;
            if (c == '}') braces--;
            if (c == '[') brackets++;
            if (c == ']') brackets--;
        }
        while (brackets > 0) { sb.append(']'); brackets--; }
        while (braces > 0) { sb.append('}'); braces--; }
        String fixed = sb.toString();
        return fixed.equals(output) ? null : fixed;
    }

    // ─────────── 降级策略：追加格式提示 ───────────

    private String buildFormatHint(ValidationResult validation, OutputFormat format) {
        if ("INCOMPLETE_DIAGNOSIS".equals(validation.issueType)) {
            return "> ⚠️ 注意：以上诊断结论可能不够完整，建议补充根因分析和具体解决方案。";
        }
        if ("MISSING_PURCHASE_FIELDS".equals(validation.issueType)) {
            return "> ⚠️ 提示：购票信息可能不完整，请确认所有必填项后再下单。";
        }
        if ("MISSING_TABLE".equals(validation.issueType)) {
            return null; // 表格缺失不追加提示，内容本身没问题
        }
        return null;
    }

    private OutputFormat detectExpectedFormat(String userText) {
        if (userText == null) return OutputFormat.NONE;
        String lower = userText.toLowerCase();

        if (lower.contains("诊断") || lower.contains("排查") || lower.contains("分析故障")) {
            return OutputFormat.DIAGNOSIS_REPORT;
        }
        if (lower.contains("买") || lower.contains("购买") || lower.contains("下单")) {
            return OutputFormat.PURCHASE_CONFIRM;
        }
        if (lower.contains("日志") || lower.contains("统计") || lower.contains("列表")) {
            return OutputFormat.TABLE;
        }
        return OutputFormat.NONE;
    }

    private ChatResponse buildRepairedResponse(ChatResponse original, String newText) {
        AssistantMessage newMsg = new AssistantMessage(newText);
        Generation newGen = new Generation(newMsg, original.getResult().getMetadata());
        return new ChatResponse(List.of(newGen), original.getMetadata());
    }

    private String extractLastUserText(List<Message> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof org.springframework.ai.chat.messages.UserMessage um) {
                return um.getText();
            }
        }
        return null;
    }

    private String truncate(String s, int max) {
        return s == null ? "" : s.length() > max ? s.substring(0, max) + "..." : s;
    }

    @Override
    public int getOrder() { return order; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private int order = Ordered.LOWEST_PRECEDENCE - 30;
        private ChatModel repairModel;

        public Builder order(int order) { this.order = order; return this; }
        public Builder repairModel(ChatModel model) { this.repairModel = model; return this; }

        public StructuredOutputAdvisor build() {
            return new StructuredOutputAdvisor(order, repairModel);
        }
    }

    private enum OutputFormat {
        NONE, DIAGNOSIS_REPORT, PURCHASE_CONFIRM, TABLE, TREE
    }

    private record ValidationResult(boolean isValid, String issueType, String issue) {}
}
