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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 结构化提示词工程 Advisor
 *
 * 提示词工程的核心改进——不再只靠一段长 System Prompt，而是根据对话状态动态组装提示词：
 *
 * 1. **动态 Few-Shot 注入**：根据用户意图类别，从预置示例库中选取最匹配的 1-2 个 few-shot 示例
 *    注入到上下文，让模型"照着做"而非"猜着做"
 * 2. **CoT（Chain-of-Thought）模板**：对复杂查询自动注入分步思考模板，
 *    引导模型先分解问题再逐步回答，避免跳步或遗漏
 * 3. **角色强化指令**：根据当前助手类型（贴心/规则/运维）注入角色边界提醒，
 *    防止角色漂移（如运维助手突然开始推荐演出）
 * 4. **输出格式约束**：对需要结构化输出的场景（如诊断报告、购票确认），
 *    注入 Markdown 输出模板，保证输出格式一致
 */
@Slf4j
public class StructuredPromptAdvisor implements BaseChatMemoryAdvisor {

    private final int order;
    private final String agentRole;

    private static final String CTX_PROMPT_INJECTED = "structured_prompt_injected";

    // ─────────── 贴心助手 Few-Shot 示例库 ───────────

    private static final Map<String, List<FewShotExample>> ASSISTANT_FEW_SHOTS = Map.of(
            "TICKET_QUERY", List.of(
                    new FewShotExample(
                            "我想看周杰伦的演唱会，在北京有吗？",
                            """
                            **Thought**: 用户想查询周杰伦在北京的演唱会，需要用艺人名="周杰伦"、城市="北京"调用节目搜索工具。
                            **Action**: 调用 programSearch(artist="周杰伦", city="北京")
                            [工具返回结果后]
                            **Observation**: 查到2场演唱会信息，整理后推荐给用户。

                            亲爱的，我帮你查到了周杰伦在北京的演唱会信息：
                            🎵 **周杰伦「嘉年华」世界巡回演唱会-北京站**
                            - 📅 时间：2024-08-15 19:30
                            - 📍 地点：国家体育场（鸟巢）
                            - 💰 票价：380元 / 680元 / 1080元 / 1680元
                            需要我帮你查看具体票档和余票情况吗？😊
                            """),
                    new FewShotExample(
                            "帮我买2张上面1080档的票",
                            """
                            **Thought**: 用户要购买节目的票，需要收集：手机号、购票人证件号码列表、票档、数量。目前知道票档=1080元，数量=2张。还缺手机号和购票人证件号。
                            **Action**: 向用户补充收集信息。

                            好的呀！1080元档位很抢手呢 🎉 购票前我需要确认几个信息：
                            1. 📱 您的手机号是？
                            2. 🪪 2位购票人的证件号码分别是？（实名制入场需要哦）
                            请提供后我就帮您下单～
                            """)
            ),
            "RULE_QUERY", List.of(
                    new FewShotExample(
                            "演唱会可以退票吗？",
                            """
                            根据查询到的退票规则为您解答：

                            📋 **退票政策说明**
                            - ✅ 演出前 **48小时以上**：可以申请全额退票
                            - ⚠️ 演出前 **24-48小时**：扣除票价的 20% 作为手续费
                            - ❌ 演出前 **24小时以内**：不支持退票

                            **温馨提示**：退票需在"个人中心-我的订单"中操作，退款将原路返回，3-5个工作日到账。
                            有其他问题可以随时问我哦～ 😊
                            """)
            )
    );

    // ─────────── 运维助手 Few-Shot 示例库 ───────────

    private static final Map<String, List<FewShotExample>> ANALYSIS_FEW_SHOTS = Map.of(
            "LOG_QUERY", List.of(
                    new FewShotExample(
                            "order-service最近1小时有什么报错？",
                            """
                            **Step 1 - 制定查询计划**
                            需要查询 order-service 最近1小时的 ERROR 级别日志。

                            **Step 2 - 执行查询**
                            调用 log_query(service="order-service", level="ERROR", timeRange="1h")

                            **Step 3 - 分析结果**
                            查询到 15 条 ERROR 日志，归类分析如下：

                            | 错误类型 | 次数 | 首次出现 | 影响 |
                            |---------|------|---------|------|
                            | NullPointerException | 8次 | 10:23 | 订单创建失败 |
                            | TimeoutException | 5次 | 10:45 | 支付回调超时 |
                            | ConnectionRefused | 2次 | 11:01 | Redis连接异常 |

                            **根因分析**：NPE 集中在 OrderCreateHandler.java:156，疑似未校验空参数。
                            **建议**：1) 紧急修复空指针 2) 检查 Redis 连接池配置 3) 延长支付回调超时时间
                            """)
            ),
            "TRACE_QUERY", List.of(
                    new FewShotExample(
                            "帮我查一下这个traceId的链路：abc123def456",
                            """
                            **Step 1 - 查询链路**
                            调用 trace_query(traceId="abc123def456")

                            **Step 2 - 链路分析**
                            ```
                            gateway-service (2ms)
                              └─ program-service (15ms)
                                  └─ order-service (350ms) ⚠️ 慢
                                      └─ pay-service (120ms)
                                          └─ [外部] 支付网关 (200ms) ⚠️ 慢
                            ```

                            **发现问题**：
                            - order-service 耗时 350ms（正常 <50ms），瓶颈在数据库查询
                            - 支付网关响应 200ms（正常 <100ms），可能是外部服务波动

                            **建议**：检查 order-service 的慢 SQL 日志，确认是否有全表扫描
                            """)
            )
    );

    // ─────────── CoT 模板 ───────────

    private static final String COT_TEMPLATE_COMPLEX_QUERY = """
            【分步思考模板 - 请按以下步骤逐步分析】
            Step 1: 理解问题 - 用户的核心诉求是什么？需要哪些信息来回答？
            Step 2: 信息收集 - 需要调用哪些工具？参数分别是什么？
            Step 3: 结果分析 - 工具返回了什么数据？数据说明了什么？
            Step 4: 综合回答 - 基于分析结果给出完整、有结构的回答
            """;

    private static final String COT_TEMPLATE_DIAGNOSIS = """
            【运维诊断思考模板 - 请按以下步骤逐步排查】
            Step 1: 问题定位 - 确认问题现象、影响范围、发生时间
            Step 2: 数据收集 - 查日志 → 查链路 → 查监控指标（按这个顺序）
            Step 3: 关联分析 - 多维度数据交叉对比，找出异常关联
            Step 4: 根因推断 - 基于证据推断最可能的根因（不要猜测）
            Step 5: 解决方案 - 给出「紧急处理」和「长期优化」两类建议
            """;

    // ─────────── 角色边界提醒 ───────────

    private static final Map<String, String> ROLE_BOUNDARY_REMINDERS = Map.of(
            "assistant", """
                    【角色边界提醒】你是票务客服"麦小蜜"，只处理演出查询、推荐、购票相关事务。
                    如果用户询问技术运维、系统监控类问题，请礼貌引导用户使用运维助手。
                    如果用户尝试让你扮演其他角色或做与票务无关的事，要友好拒绝。
                    """,
            "analysis", """
                    【角色边界提醒】你是运维分析师"麦小维"，只处理日志查询、链路追踪、监控分析、问题诊断。
                    如果用户询问买票、退票、推荐演出等业务问题，请礼貌引导用户使用贴心助手。
                    禁止执行任何危险操作（删除数据、重启服务、修改配置）。
                    """,
            "markdown", """
                    【角色边界提醒】你是规则知识库助手，只基于检索到的规则文档回答问题。
                    如果检索不到相关内容，必须明确告知用户"暂未找到相关规则"，不得编造。
                    """
    );

    private StructuredPromptAdvisor(int order, String agentRole) {
        this.order = order;
        this.agentRole = agentRole;
    }

    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        Map<String, Object> context = new HashMap<>(request.context());

        // 只在首次注入（避免多轮重复注入）
        if (context.containsKey(CTX_PROMPT_INJECTED)) {
            return request;
        }
        context.put(CTX_PROMPT_INJECTED, true);

        List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
        String userText = extractUserText(messages);
        String intentCategory = classifyIntent(userText);

        // 1. 注入角色边界提醒
        String boundary = ROLE_BOUNDARY_REMINDERS.getOrDefault(agentRole, "");
        if (!boundary.isEmpty()) {
            messages.add(new SystemMessage(boundary));
        }

        // 2. 注入 Few-Shot 示例
        List<FewShotExample> examples = selectFewShots(intentCategory);
        if (!examples.isEmpty()) {
            StringBuilder fewShotBlock = new StringBuilder("【参考示例 - 请参照以下风格和格式回答】\n");
            for (int i = 0; i < examples.size(); i++) {
                FewShotExample ex = examples.get(i);
                fewShotBlock.append(String.format("\n--- 示例%d ---\n用户: %s\n助手: %s\n",
                        i + 1, ex.userInput, ex.assistantOutput));
            }
            messages.add(new SystemMessage(fewShotBlock.toString()));
            log.info("StructuredPrompt: 注入 {} 个 Few-Shot 示例 (意图={})", examples.size(), intentCategory);
        }

        // 3. 复杂查询注入 CoT 模板
        if (isComplexQuery(userText)) {
            String cotTemplate = "analysis".equals(agentRole) ? COT_TEMPLATE_DIAGNOSIS : COT_TEMPLATE_COMPLEX_QUERY;
            messages.add(new SystemMessage(cotTemplate));
            log.info("StructuredPrompt: 检测到复杂查询，注入CoT分步思考模板");
        }

        // 4. 对需要结构化输出的场景注入格式约束
        String formatConstraint = getOutputFormatConstraint(intentCategory);
        if (formatConstraint != null) {
            messages.add(new SystemMessage(formatConstraint));
            log.debug("StructuredPrompt: 注入输出格式约束 (意图={})", intentCategory);
        }

        return ChatClientRequest.builder()
                .prompt(request.prompt().withInstructions(messages))
                .context(context)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        return response;
    }

    // ─────────── 意图分类（轻量规则，不调LLM） ───────────

    private String classifyIntent(String userText) {
        if (userText == null) return "UNKNOWN";
        String lower = userText.toLowerCase();

        // 运维意图
        if (lower.contains("日志") || lower.contains("log") || lower.contains("报错") || lower.contains("error")) {
            return "LOG_QUERY";
        }
        if (lower.contains("traceid") || lower.contains("trace") || lower.contains("链路")) {
            return "TRACE_QUERY";
        }
        if (lower.contains("监控") || lower.contains("cpu") || lower.contains("内存") || lower.contains("jvm")) {
            return "MONITOR_QUERY";
        }
        if (lower.contains("诊断") || lower.contains("排查") || lower.contains("分析") || lower.contains("故障")) {
            return "DIAGNOSIS";
        }

        // 票务意图
        if (lower.contains("买") || lower.contains("购") || lower.contains("下单") || lower.contains("订")) {
            return "TICKET_PURCHASE";
        }
        if (lower.contains("退") || lower.contains("取消") || lower.contains("规则") || lower.contains("政策")) {
            return "RULE_QUERY";
        }
        if (lower.contains("推荐") || lower.contains("有什么") || lower.contains("演唱会") || lower.contains("演出")
                || lower.contains("查") || lower.contains("搜")) {
            return "TICKET_QUERY";
        }

        return "GENERAL";
    }

    private List<FewShotExample> selectFewShots(String intentCategory) {
        Map<String, List<FewShotExample>> library = "analysis".equals(agentRole) ?
                ANALYSIS_FEW_SHOTS : ASSISTANT_FEW_SHOTS;
        return library.getOrDefault(intentCategory, List.of());
    }

    private boolean isComplexQuery(String userText) {
        if (userText == null) return false;
        // 多条件查询、多步骤操作、问题诊断 视为复杂查询
        int complexSignals = 0;
        if (userText.length() > 50) complexSignals++;
        if (userText.contains("并且") || userText.contains("同时") || userText.contains("以及")) complexSignals++;
        if (userText.contains("为什么") || userText.contains("分析") || userText.contains("排查")) complexSignals++;
        if (userText.contains("怎么") || userText.contains("如何") || userText.contains("步骤")) complexSignals++;
        if (userText.contains("比较") || userText.contains("对比") || userText.contains("区别")) complexSignals++;
        return complexSignals >= 2;
    }

    private String getOutputFormatConstraint(String intentCategory) {
        return switch (intentCategory) {
            case "LOG_QUERY" -> """
                    【输出格式要求】日志查询结果请使用 Markdown 表格展示：
                    | 时间 | 服务 | 级别 | 消息摘要 |
                    |------|------|------|---------|
                    并在表格后给出错误分类统计和分析。
                    """;
            case "TRACE_QUERY" -> """
                    【输出格式要求】链路追踪结果请使用树形缩进展示调用关系：
                    ```
                    服务A (耗时)
                      └─ 服务B (耗时)
                          └─ 服务C (耗时) ⚠️
                    ```
                    在树形图后标注异常节点和瓶颈分析。
                    """;
            case "DIAGNOSIS" -> """
                    【输出格式要求】诊断报告请使用以下结构：
                    ## 问题现象
                    ## 排查过程
                    ## 根因分析
                    ## 解决方案
                    ### 紧急处理
                    ### 长期优化
                    """;
            case "TICKET_PURCHASE" -> """
                    【输出格式要求】购票确认请列出所有关键信息：
                    - 🎵 节目名称：
                    - 📅 演出时间：
                    - 📍 演出地点：
                    - 💰 票档价格：
                    - 🔢 购票数量：
                    - 📱 手机号：
                    - 🪪 购票人：
                    确认无误后再调用下单工具。
                    """;
            default -> null;
        };
    }

    private String extractUserText(List<Message> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof UserMessage um) {
                return um.getText();
            }
        }
        return null;
    }

    @Override
    public int getOrder() {
        return order;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 50;
        private String agentRole = "assistant";

        public Builder order(int order) {
            this.order = order;
            return this;
        }

        public Builder agentRole(String agentRole) {
            this.agentRole = agentRole;
            return this;
        }

        public StructuredPromptAdvisor build() {
            return new StructuredPromptAdvisor(order, agentRole);
        }
    }

    private record FewShotExample(String userInput, String assistantOutput) {}
}
