package org.javaup.ai.advisor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 智能路由 Advisor
 * 
 * 根据用户意图自动判断应该使用哪种助手模式，并注入对应的上下文增强。
 * 作为前置路由层，可以在统一入口下自动分发到不同的处理策略。
 * 
 * 路由策略：
 * 1. 购票相关 → 注入购票流程增强指令
 * 2. 运维相关 → 注入诊断流程增强指令
 * 3. 规则查询 → 注入RAG增强指令
 * 4. 闲聊/其他 → 不注入额外指令
 * 
 * 同时根据意图复杂度动态调整工具调用策略。
 */
@Slf4j
public class AgentRouterAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final ChatModel routerModel;

    private static final String CTX_ROUTE_RESULT = "agent_route_result";
    private static final String CTX_ROUTE_CONFIDENCE = "agent_route_confidence";
    private static final String CTX_COMPLEXITY_LEVEL = "agent_complexity_level";

    private static final String ROUTER_PROMPT = """
            分析用户输入，判断其意图类别和复杂度。
            
            意图类别：
            - TICKET：购票相关（推荐节目、查询票价、下单购买、查看详情）
            - OPS：运维相关（查日志、链路追踪、监控指标、问题排查）
            - RULE：规则查询（退票规则、购票须知、优惠政策）
            - CHAT：闲聊问候（你好、谢谢、无关话题）
            
            复杂度（1-3）：
            1 = 简单（单一意图，一步完成）
            2 = 中等（需要多步操作或多个信息）
            3 = 复杂（多意图混合，需要推理和规划）
            
            用户输入：%s
            
            严格按格式输出：类别|复杂度
            例如：TICKET|2
            """;

    // 各类别的增强指令
    private static final Map<String, String> ROUTE_ENHANCEMENTS = Map.of(
            "TICKET", """
                    【路由增强-购票模式】
                    当前用户意图为购票相关。请注意：
                    - 主动收集必要信息（城市、节目类型、艺人等）
                    - 优先使用工具查询真实数据，不要编造
                    - 引导用户完成完整的购票流程
                    """,
            "OPS", """
                    【路由增强-运维模式】
                    当前用户意图为运维分析。请注意：
                    - 系统性地分析问题，先查日志再查链路
                    - 给出具体的诊断结论和解决方案
                    - 关注异常指标和错误模式
                    """,
            "RULE", """
                    【路由增强-规则查询模式】
                    当前用户意图为规则/政策查询。请注意：
                    - 优先从知识库检索准确的规则信息
                    - 引用具体条款，不要模糊回答
                    - 如果知识库中没有，明确告知用户
                    """
    );

    private AgentRouterAdvisor(int order, ChatModel routerModel) {
        this.order = order;
        this.routerModel = routerModel;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        String userInput = request.prompt().getUserMessage().getText();
        Map<String, Object> context = new HashMap<>(request.context());

        // 快速规则路由
        RouteResult result = quickRoute(userInput);

        // 无法快速判断时使用LLM路由
        if (result.category.equals("UNKNOWN") && routerModel != null) {
            result = llmRoute(userInput);
        }

        context.put(CTX_ROUTE_RESULT, result.category);
        context.put(CTX_ROUTE_CONFIDENCE, result.confidence);
        context.put(CTX_COMPLEXITY_LEVEL, result.complexity);

        log.info("智能路由: '{}' -> {} (复杂度:{}, 置信度:{})",
                truncate(userInput, 30), result.category, result.complexity, result.confidence);

        // 注入路由增强指令
        String enhancement = ROUTE_ENHANCEMENTS.get(result.category);
        if (enhancement != null) {
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.add(new SystemMessage(enhancement));

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
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        return response;
    }

    private RouteResult quickRoute(String input) {
        if (input == null || input.trim().length() < 2) {
            return new RouteResult("CHAT", 1, "high");
        }

        String lower = input.toLowerCase();

        // 运维关键词
        if (lower.contains("日志") || lower.contains("log") || lower.contains("链路") ||
            lower.contains("trace") || lower.contains("监控") || lower.contains("jvm") ||
            lower.contains("内存") || lower.contains("cpu") || lower.contains("error") && lower.contains("服务") ||
            lower.contains("报错") || lower.contains("异常") && lower.contains("排查")) {
            int complexity = (lower.contains("分析") || lower.contains("排查") || lower.contains("诊断")) ? 3 : 2;
            return new RouteResult("OPS", complexity, "high");
        }

        // 规则关键词
        if (lower.contains("退票") || lower.contains("规则") || lower.contains("须知") ||
            lower.contains("政策") || lower.contains("可以退") || lower.contains("怎么退") ||
            lower.contains("优惠") || lower.contains("注意事项")) {
            return new RouteResult("RULE", 1, "high");
        }

        // 购票关键词
        if (lower.contains("推荐") || lower.contains("买票") || lower.contains("购票") ||
            lower.contains("演唱会") || lower.contains("票价") || lower.contains("下单") ||
            lower.contains("节目") || lower.contains("演出") || lower.contains("票档")) {
            int complexity = (lower.contains("下单") || lower.contains("购买")) ? 2 : 1;
            return new RouteResult("TICKET", complexity, "high");
        }

        // 闲聊
        if (lower.matches("^(你好|hi|hello|谢谢|感谢|再见|拜拜|嗯|好的|ok|行).*")) {
            return new RouteResult("CHAT", 1, "high");
        }

        return new RouteResult("UNKNOWN", 2, "low");
    }

    private RouteResult llmRoute(String input) {
        try {
            String prompt = String.format(ROUTER_PROMPT, truncate(input, 200));
            var response = routerModel.call(new Prompt(prompt));
            String result = response.getResult().getOutput().getText().trim();

            if (result.contains("|")) {
                String[] parts = result.split("\\|");
                String category = parts[0].trim().toUpperCase();
                int complexity = parts.length > 1 ? parseComplexity(parts[1].trim()) : 2;

                if (List.of("TICKET", "OPS", "RULE", "CHAT").contains(category)) {
                    return new RouteResult(category, complexity, "medium");
                }
            }
        } catch (Exception e) {
            log.debug("LLM路由跳过: {}", e.getMessage());
        }

        return new RouteResult("TICKET", 2, "low");
    }

    private int parseComplexity(String str) {
        try {
            int val = Integer.parseInt(str.replaceAll("[^0-9]", ""));
            return Math.max(1, Math.min(3, val));
        } catch (Exception e) {
            return 2;
        }
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

    private static class RouteResult {
        final String category;
        final int complexity;
        final String confidence;

        RouteResult(String category, int complexity, String confidence) {
            this.category = category;
            this.complexity = complexity;
            this.confidence = confidence;
        }
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 5;
        private ChatModel routerModel;

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder routerModel(ChatModel routerModel) {
            this.routerModel = routerModel;
            return this;
        }

        public AgentRouterAdvisor build() {
            return new AgentRouterAdvisor(order, routerModel);
        }
    }
}
