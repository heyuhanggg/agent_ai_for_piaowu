package org.javaup.ai.config;

import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import io.modelcontextprotocol.client.McpSyncClient;

import java.util.List;


@Configuration
public class McpClientConfig {

    /**
     * 将MCP客户端的工具注册为ToolCallbackProvider
     * 这样ChatClient就可以使用MCP服务器提供的工具了
     */
    @Bean
    public ToolCallbackProvider mcpToolCallbackProvider(List<McpSyncClient> mcpSyncClients) {
        // 把所有已连接的 MCP Client 包装成 Spring AI 可识别的 ToolCallbackProvider。
        // 运维助手拿到这个 provider 后，就能在对话过程中直接调用 MCP 服务端暴露的日志查询工具。
        return new SyncMcpToolCallbackProvider(mcpSyncClients);
    }
}
