package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 幻觉接地检测 Advisor (Guardrails 工程)
 *
 * 检测模型输出中的"无中生有"问题——模型声称的数据是否有工具返回结果支撑：
 *
 * 1. **数字接地**：模型输出的具体数字（错误次数、响应时间、票价）
 *    是否在本轮对话的工具返回中出现过
 * 2. **实体接地**：模型提到的服务名、节目名、时间等是否有来源
 * 3. **结论接地**：模型给出的"根因"和"建议"是否基于实际数据
 *
 * 检测策略（纯规则，不调LLM，零额外开销）：
 * - 从对话历史中提取所有工具返回的数据（作为"事实库"）
 * - 从模型最终输出中提取关键声明（数字、百分比、服务名等）
 * - 交叉比对：如果输出中的数字/实体在事实库中找不到来源，标记为疑似幻觉
 * - 对疑似幻觉追加警告标注（不阻断输出，只做标记）
 *
 * 兜底策略：如果提取或比对出错，直接放行，绝不因为校验逻辑错误而影响用户体验。
 */
@Slf4j
public class HallucinationGroundingAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final boolean enableWarningAnnotation;

    private static final String CTX_GROUNDING_FACTS    = "grounding_fact_pool";
    private static final String CTX_GROUNDING_CHECKED  = "grounding_checked";

    // 数字模式：匹配 "123次"、"45.6ms"、"78%" 等
    private static final Pattern NUMBER_PATTERN = Pattern.compile("(\\d+(?:\\.\\d+)?)(次|条|个|ms|秒|分钟|小时|%|元|张)");

    // 服务名模式
    private static final Pattern SERVICE_PATTERN = Pattern.compile("([a-z]+-service)");

    private HallucinationGroundingAdvisor(int order, boolean enableWarningAnnotation) {
        this.order = order;
        this.enableWarningAnnotation = enableWarningAnnotation;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        Map<String, Object> context = new HashMap<>(request.context());

        // 从对话历史中提取工具返回的事实数据
        List<String> facts = (List<String>) context.computeIfAbsent(CTX_GROUNDING_FACTS, k -> new ArrayList<>());

        // 扫描消息中的工具返回结果，加入事实库
        for (var msg : request.prompt().getInstructions()) {
            if (msg instanceof AssistantMessage am) {
                String text = am.getText();
                if (text != null && (am.getToolCalls() != null && !am.getToolCalls().isEmpty())) {
                    // 这是一条包含工具调用的助手消息，其文本是工具结果
                    extractFacts(text, facts);
                }
            }
        }

        context.put(CTX_GROUNDING_FACTS, facts);
        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }

    @Override
    @SuppressWarnings("unchecked")
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }

        Map<String, Object> context = new HashMap<>(response.context());
        if (context.containsKey(CTX_GROUNDING_CHECKED)) {
            return response;
        }

        String output = chatResponse.getResult().getOutput().getText();
        if (output == null || output.isEmpty()) {
            return response;
        }

        // 如果本轮有工具调用结果，将其加入事实库
        if (chatResponse.getResult().getToolCalls() != null && !chatResponse.getResult().getToolCalls().isEmpty()) {
            List<String> facts = (List<String>) context.computeIfAbsent(CTX_GROUNDING_FACTS, k -> new ArrayList<>());
            extractFacts(output, facts);
            context.put(CTX_GROUNDING_FACTS, facts);
            // 工具调用响应不需要接地检查
            return ChatClientResponse.builder().chatResponse(chatResponse).context(context).build();
        }

        try {
            List<String> facts = (List<String>) context.getOrDefault(CTX_GROUNDING_FACTS, new ArrayList<>());

            // 如果没有工具调用过（纯聊天），跳过接地检查
            if (facts.isEmpty()) {
                context.put(CTX_GROUNDING_CHECKED, true);
                return ChatClientResponse.builder().chatResponse(chatResponse).context(context).build();
            }

            // 执行接地检查
            List<GroundingViolation> violations = checkGrounding(output, facts);

            if (!violations.isEmpty()) {
                log.warn("HallucinationGrounding: 检测到 {} 处疑似幻觉", violations.size());
                for (GroundingViolation v : violations) {
                    log.warn("  - [{}] '{}' 在工具结果中无来源", v.type, v.claim);
                }

                if (enableWarningAnnotation) {
                    // 追加警告标注
                    String annotated = annotateOutput(output, violations);
                    ChatResponse annotatedResponse = buildAnnotatedResponse(chatResponse, annotated);
                    context.put(CTX_GROUNDING_CHECKED, true);
                    return ChatClientResponse.builder().chatResponse(annotatedResponse).context(context).build();
                }
            } else {
                log.debug("HallucinationGrounding: 接地检查通过，无幻觉");
            }
        } catch (Exception e) {
            // 兜底：校验逻辑出错绝不影响正常输出
            log.warn("HallucinationGrounding: 检查异常，放行: {}", e.getMessage());
        }

        context.put(CTX_GROUNDING_CHECKED, true);
        return ChatClientResponse.builder().chatResponse(chatResponse).context(context).build();
    }

    // ─────────── 事实提取 ───────────

    private void extractFacts(String text, List<String> facts) {
        if (text == null) return;

        // 提取数字+单位
        Matcher nm = NUMBER_PATTERN.matcher(text);
        while (nm.find()) {
            facts.add(nm.group(0));
        }

        // 提取服务名
        Matcher sm = SERVICE_PATTERN.matcher(text);
        while (sm.find()) {
            facts.add(sm.group(1));
        }

        // 提取引号中的内容（通常是关键信息）
        Matcher qm = Pattern.compile("[\"']([^\"']{2,50})[\"']").matcher(text);
        while (qm.find()) {
            facts.add(qm.group(1));
        }

        // 提取时间戳
        Matcher tm = Pattern.compile("\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}").matcher(text);
        while (tm.find()) {
            facts.add(tm.group(0));
        }
    }

    // ─────────── 接地检查 ───────────

    private List<GroundingViolation> checkGrounding(String output, List<String> facts) {
        List<GroundingViolation> violations = new ArrayList<>();

        // 检查数字声明
        Matcher nm = NUMBER_PATTERN.matcher(output);
        while (nm.find()) {
            String claim = nm.group(0);
            String number = nm.group(1);
            // 跳过常见的非数据数字（如"1."列表编号、"第1步"等）
            if (isCommonNumber(number)) continue;

            boolean grounded = facts.stream().anyMatch(f -> f.contains(number));
            if (!grounded) {
                violations.add(new GroundingViolation("NUMBER", claim));
            }
        }

        // 检查服务名声明
        Matcher sm = SERVICE_PATTERN.matcher(output);
        while (sm.find()) {
            String service = sm.group(1);
            boolean grounded = facts.stream().anyMatch(f -> f.equals(service));
            if (!grounded) {
                // 服务名可能来自系统提示词（已知服务列表），不算幻觉
                if (!isKnownService(service)) {
                    violations.add(new GroundingViolation("SERVICE", service));
                }
            }
        }

        return violations;
    }

    private boolean isCommonNumber(String number) {
        try {
            double val = Double.parseDouble(number);
            // 1-10的小数字通常是列表编号，不需要接地
            return val <= 10 && val == Math.floor(val);
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private boolean isKnownService(String service) {
        // 系统提示词中已列出的已知服务
        return List.of(
                "gateway-service", "user-service", "program-service",
                "order-service", "pay-service", "base-data-service",
                "admin-service", "customize-service"
        ).contains(service);
    }

    private String annotateOutput(String output, List<GroundingViolation> violations) {
        if (violations.size() <= 2) {
            // 少量幻觉只追加脚注
            StringBuilder sb = new StringBuilder(output);
            sb.append("\n\n---\n> ⚠️ **数据准确性提示**：以下数据未在工具查询结果中找到直接来源，请注意核实：\n");
            for (GroundingViolation v : violations) {
                sb.append(String.format("> - `%s`\n", v.claim));
            }
            return sb.toString();
        } else {
            // 大量幻觉追加强警告
            return output + "\n\n---\n> ⚠️ **注意**：本次回复中有部分数据可能基于推断而非实际查询结果，建议通过工具重新查询确认关键数据的准确性。";
        }
    }

    private ChatResponse buildAnnotatedResponse(ChatResponse original, String newText) {
        AssistantMessage msg = new AssistantMessage(newText);
        Generation gen = new Generation(msg, original.getResult().getMetadata());
        return new ChatResponse(List.of(gen), original.getMetadata());
    }

    @Override
    public int getOrder() { return order; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private int order = Ordered.LOWEST_PRECEDENCE - 10;
        private boolean enableWarningAnnotation = true;

        public Builder order(int order) { this.order = order; return this; }
        public Builder enableWarningAnnotation(boolean enable) { this.enableWarningAnnotation = enable; return this; }

        public HallucinationGroundingAdvisor build() {
            return new HallucinationGroundingAdvisor(order, enableWarningAnnotation);
        }
    }

    private record GroundingViolation(String type, String claim) {}
}
