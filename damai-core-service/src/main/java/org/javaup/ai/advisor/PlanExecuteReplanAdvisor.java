package org.javaup.ai.advisor;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseChatMemoryAdvisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.ToolCall;
import org.springframework.core.Ordered;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
public class PlanExecuteReplanAdvisor implements BaseChatMemoryAdvisor {
    
    private final int order;
    private final int maxReplans;
    private final boolean enablePlanning;
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    private static final String CTX_PLAN_STATE = "plan_state";
    private static final String CTX_EXECUTION_STEPS = "execution_steps";
    private static final String CTX_REPLAN_COUNT = "replan_count";
    private static final String CTX_CURRENT_STEP = "current_step_index";
    
    private static final String PLAN_SYSTEM_PROMPT = """
            你是一个使用Plan-Execute-Replan模式的智能运维助手。
            
            工作流程：
            1. **Plan (制定计划)**: 分析问题，制定详细的诊断步骤计划
            2. **Execute (执行)**: 按步骤调用工具，收集信息
            3. **Observe (观察)**: 分析执行结果
            4. **Replan (重新规划)**: 根据结果动态调整后续计划
            
            计划格式（JSON）：
            {
              "goal": "诊断目标",
              "steps": [
                {"id": 1, "action": "查询错误日志", "tool": "log_query", "status": "pending"},
                {"id": 2, "action": "追踪请求链路", "tool": "trace_query", "status": "pending"},
                {"id": 3, "action": "分析根因", "tool": "analyze", "status": "pending"}
              ]
            }
            
            重要规则：
            - 首次回答时，先输出完整的诊断计划（JSON格式）
            - 执行每步后评估结果，决定是否需要调整计划
            - 如发现新线索，可以动态添加或修改步骤
            - 最多重新规划 %d 次
            - 完成所有步骤后给出诊断结论和建议
            """;
    
    private PlanExecuteReplanAdvisor(int order, int maxReplans, boolean enablePlanning) {
        this.order = order;
        this.maxReplans = maxReplans;
        this.enablePlanning = enablePlanning;
    }
    
    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        if (!enablePlanning) {
            return request;
        }
        
        Map<String, Object> context = new HashMap<>(request.context());
        
        // 初始化计划状态
        if (!context.containsKey(CTX_PLAN_STATE)) {
            context.put(CTX_PLAN_STATE, PlanState.PLANNING);
            context.put(CTX_EXECUTION_STEPS, new ArrayList<ExecutionStep>());
            context.put(CTX_REPLAN_COUNT, 0);
            context.put(CTX_CURRENT_STEP, -1);
            
            log.info("Plan-Execute-Replan模式已启用，最大重规划次数: {}", maxReplans);
            
            // 注入计划系统提示词
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.add(0, new SystemMessage(String.format(PLAN_SYSTEM_PROMPT, maxReplans)));
            
            // 添加引导消息让模型输出计划
            String userMsg = request.prompt().getUserMessage().getText();
            String guidedMsg = String.format("""
                    用户问题: %s
                    
                    请先制定诊断计划（JSON格式），然后开始执行第一步。
                    """, userMsg);
            
            messages.add(new UserMessage(guidedMsg));
            
            return ChatClientRequest.builder()
                    .prompt(request.prompt().withInstructions(messages))
                    .context(context)
                    .build();
        }
        
        // 获取当前状态
        PlanState state = (PlanState) context.get(CTX_PLAN_STATE);
        Integer replanCount = (Integer) context.get(CTX_REPLAN_COUNT);
        
        if (state == PlanState.REPLANNING && replanCount >= maxReplans) {
            log.warn("已达到最大重规划次数: {}", maxReplans);
            context.put(CTX_PLAN_STATE, PlanState.FINALIZING);
        }
        
        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        if (!enablePlanning) {
            return response;
        }
        
        Map<String, Object> context = response.context();
        PlanState state = (PlanState) context.getOrDefault(CTX_PLAN_STATE, PlanState.PLANNING);
        List<ExecutionStep> steps = (List<ExecutionStep>) context.getOrDefault(
                CTX_EXECUTION_STEPS, new ArrayList<>());
        Integer currentStepIndex = (Integer) context.getOrDefault(CTX_CURRENT_STEP, -1);
        Integer replanCount = (Integer) context.getOrDefault(CTX_REPLAN_COUNT, 0);
        
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }
        
        String assistantOutput = chatResponse.getResult().getOutput().getText();
        List<ToolCall> toolCalls = chatResponse.getResult().getToolCalls();
        
        switch (state) {
            case PLANNING:
                // 尝试解析计划
                List<ExecutionStep> parsedSteps = extractPlan(assistantOutput);
                if (!parsedSteps.isEmpty()) {
                    steps.addAll(parsedSteps);
                    context.put(CTX_EXECUTION_STEPS, steps);
                    context.put(CTX_PLAN_STATE, PlanState.EXECUTING);
                    context.put(CTX_CURRENT_STEP, 0);
                    log.info("计划制定完成，共 {} 个步骤", steps.size());
                    logPlan(steps);
                }
                break;
                
            case EXECUTING:
                // 记录工具调用
                if (toolCalls != null && !toolCalls.isEmpty() && currentStepIndex < steps.size()) {
                    ExecutionStep currentStep = steps.get(currentStepIndex);
                    currentStep.setStatus(StepStatus.COMPLETED);
                    currentStep.setToolCalls(toolCalls.stream()
                            .map(ToolCall::name)
                            .collect(Collectors.toList()));
                    currentStep.setResult(assistantOutput);
                    
                    log.info("步骤 {}/{} 执行完成: {}", currentStepIndex + 1, steps.size(), 
                            currentStep.getAction());
                    
                    // 检查是否需要重新规划
                    if (shouldReplan(assistantOutput) && replanCount < maxReplans) {
                        context.put(CTX_PLAN_STATE, PlanState.REPLANNING);
                        context.put(CTX_REPLAN_COUNT, replanCount + 1);
                        log.info("触发重新规划 ({}/{})", replanCount + 1, maxReplans);
                    } else {
                        // 继续下一步
                        int nextStep = currentStepIndex + 1;
                        if (nextStep < steps.size()) {
                            context.put(CTX_CURRENT_STEP, nextStep);
                            log.info("准备执行步骤 {}/{}: {}", nextStep + 1, steps.size(), 
                                    steps.get(nextStep).getAction());
                        } else {
                            context.put(CTX_PLAN_STATE, PlanState.FINALIZING);
                            log.info("所有步骤执行完成，准备输出最终结论");
                        }
                    }
                }
                break;
                
            case REPLANNING:
                // 解析新增或修改的步骤
                List<ExecutionStep> newSteps = extractPlan(assistantOutput);
                if (!newSteps.isEmpty()) {
                    // 将新步骤插入到当前位置之后
                    int insertPos = currentStepIndex + 1;
                    for (ExecutionStep newStep : newSteps) {
                        if (newStep.getStatus() == StepStatus.PENDING) {
                            steps.add(insertPos++, newStep);
                        }
                    }
                    log.info("计划已更新，新增 {} 个步骤，当前共 {} 个步骤", 
                            newSteps.size(), steps.size());
                    logPlan(steps);
                }
                context.put(CTX_PLAN_STATE, PlanState.EXECUTING);
                break;
                
            case FINALIZING:
                log.info("诊断流程完成");
                logExecutionSummary(steps);
                break;
        }
        
        return ChatClientResponse.builder()
                .chatResponse(chatResponse)
                .context(context)
                .build();
    }
    
    private List<ExecutionStep> extractPlan(String output) {
        List<ExecutionStep> steps = new ArrayList<>();
        
        try {
            // 尝试从JSON中提取计划
            if (output.contains("\"steps\"") && output.contains("[")) {
                int start = output.indexOf("[", output.indexOf("\"steps\""));
                int end = findMatchingBracket(output, start);
                if (end > start) {
                    String jsonArray = output.substring(start, end + 1);
                    List<Map<String, Object>> rawSteps = objectMapper.readValue(
                            jsonArray, new TypeReference<List<Map<String, Object>>>() {});
                    
                    for (Map<String, Object> rawStep : rawSteps) {
                        ExecutionStep step = new ExecutionStep();
                        step.setId(((Number) rawStep.getOrDefault("id", steps.size() + 1)).intValue());
                        step.setAction((String) rawStep.get("action"));
                        step.setTool((String) rawStep.get("tool"));
                        step.setStatus(StepStatus.PENDING);
                        steps.add(step);
                    }
                }
            }
        } catch (Exception e) {
            log.debug("无法解析计划JSON，使用文本解析: {}", e.getMessage());
            // 降级方案：从文本中提取步骤
            steps.addAll(extractStepsFromText(output));
        }
        
        return steps;
    }
    
    private List<ExecutionStep> extractStepsFromText(String text) {
        List<ExecutionStep> steps = new ArrayList<>();
        String[] lines = text.split("\n");
        
        for (String line : lines) {
            if (line.matches(".*\\d+[.、].*") || line.contains("步骤")) {
                ExecutionStep step = new ExecutionStep();
                step.setId(steps.size() + 1);
                step.setAction(line.trim());
                step.setStatus(StepStatus.PENDING);
                steps.add(step);
            }
        }
        
        return steps;
    }
    
    private int findMatchingBracket(String str, int start) {
        int count = 1;
        for (int i = start + 1; i < str.length(); i++) {
            if (str.charAt(i) == '[') count++;
            if (str.charAt(i) == ']') count--;
            if (count == 0) return i;
        }
        return -1;
    }
    
    private boolean shouldReplan(String output) {
        if (output == null) return false;
        String lowerOutput = output.toLowerCase();
        
        // 显式重规划信号
        if (lowerOutput.contains("需要调整") || lowerOutput.contains("发现新线索") ||
            lowerOutput.contains("重新规划") || lowerOutput.contains("补充步骤") ||
            lowerOutput.contains("新发现") || lowerOutput.contains("意外")) {
            return true;
        }
        
        // 失败信号 - 工具调用返回错误
        if (lowerOutput.contains("失败") || lowerOutput.contains("error") ||
            lowerOutput.contains("exception") || lowerOutput.contains("超时") ||
            lowerOutput.contains("timeout") || lowerOutput.contains("无法连接")) {
            log.info("检测到执行失败信号，触发重规划");
            return true;
        }
        
        return false;
    }
    
    private void logPlan(List<ExecutionStep> steps) {
        StringBuilder sb = new StringBuilder("\n=== 执行计划 ===\n");
        for (int i = 0; i < steps.size(); i++) {
            ExecutionStep step = steps.get(i);
            sb.append(String.format("%d. [%s] %s\n", i + 1, step.getStatus(), step.getAction()));
        }
        sb.append("===============");
        log.info(sb.toString());
    }
    
    private void logExecutionSummary(List<ExecutionStep> steps) {
        long completed = steps.stream().filter(s -> s.getStatus() == StepStatus.COMPLETED).count();
        long failed = steps.stream().filter(s -> s.getStatus() == StepStatus.FAILED).count();
        long skipped = steps.stream().filter(s -> s.getStatus() == StepStatus.SKIPPED).count();
        long pending = steps.stream().filter(s -> s.getStatus() == StepStatus.PENDING).count();
        int totalRetries = steps.stream().mapToInt(ExecutionStep::getRetryCount).sum();
        
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("\n=== 执行摘要 ===\n"));
        sb.append(String.format("完成: %d | 失败: %d | 跳过: %d | 未执行: %d | 重试总次数: %d\n", 
                completed, failed, skipped, pending, totalRetries));
        for (ExecutionStep step : steps) {
            String emoji = switch (step.getStatus()) {
                case COMPLETED -> "✓";
                case FAILED -> "✗";
                case SKIPPED -> "→";
                default -> "○";
            };
            sb.append(String.format("  %s [%d] %s (重试:%d)\n", emoji, step.getId(), 
                    step.getAction(), step.getRetryCount()));
        }
        sb.append("===============");
        log.info(sb.toString());
    }
    
    @Override
    public int getOrder() {
        return order;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static final class Builder {
        private int order = Ordered.HIGHEST_PRECEDENCE + 100;
        private int maxReplans = 3;
        private boolean enablePlanning = true;
        
        public Builder order(int order) {
            this.order = order;
            return this;
        }
        
        public Builder maxReplans(int maxReplans) {
            this.maxReplans = maxReplans;
            return this;
        }
        
        public Builder enablePlanning(boolean enablePlanning) {
            this.enablePlanning = enablePlanning;
            return this;
        }
        
        public PlanExecuteReplanAdvisor build() {
            return new PlanExecuteReplanAdvisor(order, maxReplans, enablePlanning);
        }
    }
    
    private enum PlanState {
        PLANNING,      // 制定计划中
        EXECUTING,     // 执行步骤中
        REPLANNING,    // 重新规划中
        FINALIZING     // 输出结论中
    }
    
    private enum StepStatus {
        PENDING,       // 待执行
        EXECUTING,     // 执行中
        COMPLETED,     // 已完成
        FAILED,        // 执行失败
        RETRYING,      // 重试中
        SKIPPED        // 已跳过
    }
    
    private static class ExecutionStep {
        private int id;
        private String action;
        private String tool;
        private StepStatus status;
        private List<String> toolCalls;
        private String result;
        private int retryCount;
        private List<Integer> dependsOn;
        
        public int getId() { return id; }
        public void setId(int id) { this.id = id; }
        
        public String getAction() { return action; }
        public void setAction(String action) { this.action = action; }
        
        public String getTool() { return tool; }
        public void setTool(String tool) { this.tool = tool; }
        
        public StepStatus getStatus() { return status; }
        public void setStatus(StepStatus status) { this.status = status; }
        
        public List<String> getToolCalls() { return toolCalls; }
        public void setToolCalls(List<String> toolCalls) { this.toolCalls = toolCalls; }
        
        public String getResult() { return result; }
        public void setResult(String result) { this.result = result; }
        
        public int getRetryCount() { return retryCount; }
        public void setRetryCount(int retryCount) { this.retryCount = retryCount; }
        
        public List<Integer> getDependsOn() { return dependsOn; }
        public void setDependsOn(List<Integer> dependsOn) { this.dependsOn = dependsOn; }
    }
}
