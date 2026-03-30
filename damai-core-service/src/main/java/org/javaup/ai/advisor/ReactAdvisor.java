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
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.ToolCall;
import org.springframework.core.Ordered;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
public class ReactAdvisor implements BaseChatMemoryAdvisor {
    
    private final int order;
    private final int maxIterations;
    private final boolean enableReactLoop;
    
    private static final String CTX_REACT_ITERATION = "react_iteration_count";
    private static final String CTX_REACT_HISTORY = "react_thought_history";
    private static final String CTX_REACT_ENABLED = "react_loop_enabled";
    
    private static final String REACT_SYSTEM_PROMPT = """
            你是一个使用ReAct (Reason + Act) 决策模式的智能助手。
            
            工作流程：
            1. **Thought (思考)**: 分析当前情况，思考下一步该做什么
            2. **Action (行动)**: 调用工具执行具体操作
            3. **Observation (观察)**: 观察工具返回的结果
            4. **重复**: 根据观察结果继续思考，直到得出最终答案
            
            重要规则：
            - 每次回答时先说明你的思考过程（以"思考："开头）
            - 如需调用工具，明确说明调用原因
            - 观察工具结果后，评估是否需要进一步操作
            - 当收集到足够信息后，给出最终答案
            - 最多进行 %d 轮迭代
            """;
    
    private ReactAdvisor(int order, int maxIterations, boolean enableReactLoop) {
        this.order = order;
        this.maxIterations = maxIterations;
        this.enableReactLoop = enableReactLoop;
    }
    
    @Override
    public ChatClientRequest before(ChatClientRequest request, AdvisorChain chain) {
        if (!enableReactLoop) {
            return request;
        }
        
        Map<String, Object> context = new HashMap<>(request.context());
        
        // 初始化React上下文
        if (!context.containsKey(CTX_REACT_ITERATION)) {
            context.put(CTX_REACT_ITERATION, 0);
            context.put(CTX_REACT_HISTORY, new ArrayList<String>());
            context.put(CTX_REACT_ENABLED, true);
            
            log.info("React模式已启用，最大迭代次数: {}", maxIterations);
            
            // 注入React系统提示词
            List<Message> messages = new ArrayList<>(request.prompt().getInstructions());
            messages.add(0, new SystemMessage(String.format(REACT_SYSTEM_PROMPT, maxIterations)));
            
            return ChatClientRequest.builder()
                    .prompt(request.prompt().withInstructions(messages))
                    .context(context)
                    .build();
        }
        
        // 检查迭代次数
        Integer iteration = (Integer) context.get(CTX_REACT_ITERATION);
        if (iteration >= maxIterations) {
            log.warn("React迭代已达到最大次数: {}", maxIterations);
            context.put(CTX_REACT_ENABLED, false);
        }
        
        return ChatClientRequest.builder()
                .prompt(request.prompt())
                .context(context)
                .build();
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public ChatClientResponse after(ChatClientResponse response, AdvisorChain chain) {
        if (!enableReactLoop) {
            return response;
        }
        
        Map<String, Object> context = response.context();
        Integer iteration = (Integer) context.getOrDefault(CTX_REACT_ITERATION, 0);
        List<String> thoughtHistory = (List<String>) context.getOrDefault(CTX_REACT_HISTORY, new ArrayList<>());
        Boolean reactEnabled = (Boolean) context.getOrDefault(CTX_REACT_ENABLED, false);
        
        if (!reactEnabled || iteration >= maxIterations) {
            return response;
        }
        
        ChatResponse chatResponse = response.chatResponse();
        if (chatResponse == null || chatResponse.getResult() == null) {
            return response;
        }
        
        String assistantOutput = chatResponse.getResult().getOutput().getText();
        List<ToolCall> toolCalls = chatResponse.getResult().getToolCalls();
        
        // 记录思考过程
        String thoughtRecord = String.format("[迭代 %d] 思考: %s", iteration + 1, 
                                            extractThought(assistantOutput));
        thoughtHistory.add(thoughtRecord);
        
        // 检查是否有工具调用
        if (toolCalls != null && !toolCalls.isEmpty()) {
            log.info("React迭代 {}/{}: 模型决定调用 {} 个工具", 
                    iteration + 1, maxIterations, toolCalls.size());
            
            for (ToolCall toolCall : toolCalls) {
                String actionRecord = String.format("[迭代 %d] 行动: 调用工具 %s", 
                                                   iteration + 1, toolCall.name());
                thoughtHistory.add(actionRecord);
            }
            
            // 更新迭代计数
            context.put(CTX_REACT_ITERATION, iteration + 1);
            context.put(CTX_REACT_HISTORY, thoughtHistory);
        } else {
            // 没有工具调用，说明模型认为可以给出最终答案
            log.info("React迭代 {}/{}: 模型给出最终答案", iteration + 1, maxIterations);
            context.put(CTX_REACT_ENABLED, false);
            
            // 在最终答案中添加思考历史摘要
            if (!thoughtHistory.isEmpty()) {
                log.debug("React思考历史:\n{}", String.join("\n", thoughtHistory));
            }
        }
        
        return ChatClientResponse.builder()
                .chatResponse(chatResponse)
                .context(context)
                .build();
    }
    
    private String extractThought(String output) {
        if (output == null) {
            return "无";
        }
        
        // 提取"思考："部分
        if (output.contains("思考：") || output.contains("思考:")) {
            String[] parts = output.split("思考：|思考:");
            if (parts.length > 1) {
                String thought = parts[1].split("\n")[0];
                return thought.length() > 100 ? thought.substring(0, 100) + "..." : thought;
            }
        }
        
        return output.length() > 100 ? output.substring(0, 100) + "..." : output;
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
        private int maxIterations = 5;
        private boolean enableReactLoop = true;
        
        public Builder order(int order) {
            this.order = order;
            return this;
        }
        
        public Builder maxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }
        
        public Builder enableReactLoop(boolean enableReactLoop) {
            this.enableReactLoop = enableReactLoop;
            return this;
        }
        
        public ReactAdvisor build() {
            return new ReactAdvisor(order, maxIterations, enableReactLoop);
        }
    }
}
