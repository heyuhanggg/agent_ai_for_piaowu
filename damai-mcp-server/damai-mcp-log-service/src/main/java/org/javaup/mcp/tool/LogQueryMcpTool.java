package org.javaup.mcp.tool;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.dromara.easyes.core.conditions.select.LambdaEsQueryWrapper;
import org.javaup.mcp.entity.LogDocument;
import org.javaup.mcp.mapper.LogMapper;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


@Slf4j
@Service
@RequiredArgsConstructor
public class LogQueryMcpTool {

    // ES 日志访问入口，负责真正执行日志查询。
    private final LogMapper logMapper;
    
    // JSON 解析器，主要用于解析 ES 聚合查询返回结果。
    private final ObjectMapper objectMapper;
    
    /**
     * 获取可用的服务列表
     */
    // 让运维助手先知道系统里有哪些微服务，便于后续引导用户选服务或自动补全服务名。
    @Tool(description = "获取演出票务系统中所有可用的微服务列表")
    public ToolResult getServiceList() {
        // 先从 ES 聚合查询里拿到服务名列表。
        List<String> serviceList = getServiceListFromEs();
        
        // 统一封装返回数据，便于模型读取和总结。
        Map<String, Object> data = new HashMap<>();
        data.put("服务列表", serviceList);
        data.put("服务数量", serviceList.size());
        return ToolResult.success("获取服务列表成功", data);
    }
    
    /**
     * 从 ES 中获取服务列表（使用 DSL 聚合查询）
     */
    private List<String> getServiceListFromEs() {
        try {
            // 这里直接构造原生 ES DSL 聚合查询。
            // 目的不是查日志明细，而是聚合 projectName.keyword 字段，得到全量服务名列表。
            String dsl = "{" +
                    "\"size\": 0," +
                    "\"aggs\": {" +
                    "  \"service_names\": {" +
                    "    \"terms\": {" +
                    "      \"field\": \"projectName.keyword\"," +
                    "      \"size\": 100" +
                    "    }" +
                    "  }" +
                    "}" +
                    "}";
            // 执行 DSL 并拿到原始 JSON 结果。
            String jsonResult = logMapper.executeDSL(dsl);
            // 再从 JSON 中提取 buckets 里的服务名。
            return parseServiceListFromJson(jsonResult);
        } catch (Exception e) {
            // 查询失败时记录日志，并返回空列表，避免工具直接崩溃。
            log.error("DSL查询服务列表失败，使用默认列表", e);
            return new ArrayList<>();
        }
    }
    
    /**
     * 从 JSON 结果中解析服务列表
     */
    private List<String> parseServiceListFromJson(String jsonResult) {
        List<String> serviceList = new ArrayList<>();
        try {
            // 解析 ES 返回的聚合 JSON 结构。
            JsonNode root = objectMapper.readTree(jsonResult);
            JsonNode buckets = root.path("aggregations").path("service_names").path("buckets");
            if (buckets.isArray()) {
                // 遍历每个 bucket，取出服务名 key。
                for (JsonNode bucket : buckets) {
                    String key = bucket.path("key").asText();
                    if (key != null && !key.isEmpty()) {
                        serviceList.add(key);
                    }
                }
            }
        } catch (Exception e) {
            log.error("解析服务列表JSON失败", e);
        }
        // 最后排序后返回，保证前端展示和模型阅读都更稳定。
        Collections.sort(serviceList);
        return serviceList;
    }

    /**
     * 按关键词搜索日志
     */
    // 这是运维助手里最常用的工具之一：按关键字搜日志，可选过滤服务名和日志级别。
    @Tool(description = "根据关键词搜索日志内容，支持模糊匹配日志消息")
    public ToolResult searchLogsByKeyword(
            @ToolParam(description = "搜索关键词，用于匹配日志消息内容") String keyword,
            @ToolParam(description = "服务名称，可选。如：gateway-service、order-service等", required = false) String serviceName,
            @ToolParam(description = "日志级别，可选。如：INFO、WARN、ERROR、DEBUG", required = false) String level,
            @ToolParam(description = "返回的日志条数，默认20条", required = false) Integer size) {

        try {
            // 对返回条数做保护，避免一次取太多日志导致模型上下文过大。
            int limit = (size != null && size > 0) ? Math.min(size, 100) : 20;

            // 构造 ES 查询条件，核心条件是 message 模糊匹配关键词。
            LambdaEsQueryWrapper<LogDocument> wrapper = new LambdaEsQueryWrapper<>();
            wrapper.match(LogDocument::getMessage, keyword);

            // 如果指定了服务名，则进一步按服务过滤。
            if (serviceName != null && !serviceName.isEmpty()) {
                wrapper.match(LogDocument::getProjectName, serviceName);
            }
            // 如果指定了级别，则按日志级别过滤，例如 ERROR/WARN。
            if (level != null && !level.isEmpty()) {
                wrapper.match(LogDocument::getLevel, level.toUpperCase());
            }

            // 按时间倒序，优先返回最新日志。
            wrapper.orderByDesc(LogDocument::getTimestamp);
            wrapper.limit(limit);

            // 执行查询。
            List<LogDocument> logs = logMapper.selectList(wrapper);

            // 把日志结果和查询条件封装成统一结构返回给模型。
            Map<String, Object> data = new HashMap<>();
            data.put("查询条件", buildQueryDesc(keyword, serviceName, level));
            data.put("日志数量", logs.size());
            data.put("日志列表", formatLogs(logs));

            return ToolResult.success("日志搜索成功", data);
        } catch (Exception e) {
            log.error("日志搜索失败", e);
            return ToolResult.error("日志搜索失败: " + e.getMessage());
        }
    }

    /**
     * 通过 traceId 查询调用链路日志
     */
    // traceId 链路追踪工具：适合排查一次请求跨多个微服务的调用路径。
    @Tool(description = "通过traceId查询完整的调用链路日志，串联所有微服务的日志记录，用于问题排查和链路追踪")
    public ToolResult getLogsByTraceId(
            @ToolParam(description = "链路追踪ID（traceId）") String traceId) {

        try {
            // 先做基础参数校验，避免无效 traceId 打到 ES。
            if (traceId == null || traceId.isEmpty() || "-".equals(traceId)) {
                return ToolResult.error("请提供有效的traceId");
            }

            // 按 traceId 精确查询该链路下的所有日志。
            LambdaEsQueryWrapper<LogDocument> wrapper = new LambdaEsQueryWrapper<>();
            wrapper.match(LogDocument::getTraceId, traceId);
            // 按时间正序排序，便于还原真实调用顺序。
            wrapper.orderByAsc(LogDocument::getTimeMillis);
            wrapper.limit(200);

            List<LogDocument> logs = logMapper.selectList(wrapper);

            // 没查到日志时直接返回错误结果，提示 traceId 不存在或日志未入库。
            if (logs.isEmpty()) {
                return ToolResult.error("未找到traceId为 " + traceId + " 的日志记录");
            }

            // 按服务分组，便于模型分析每个服务在链路中输出了哪些日志。
            Map<String, List<Map<String, Object>>> logsByService = logs.stream()
                    .collect(Collectors.groupingBy(
                            LogDocument::getProjectName,
                            LinkedHashMap::new,
                            Collectors.mapping(this::formatLog, Collectors.toList())
                    ));

            // 生成涉及服务的顺序列表，帮助模型总结请求依次经过了哪些服务。
            List<String> serviceOrder = logs.stream()
                    .map(LogDocument::getProjectName)
                    .distinct()
                    .collect(Collectors.toList());

            // 把 traceId、日志总量、服务顺序和分组日志统一返回。
            Map<String, Object> data = new HashMap<>();
            data.put("traceId", traceId);
            data.put("日志总数", logs.size());
            data.put("涉及服务", serviceOrder);
            data.put("调用链路", logsByService);

            return ToolResult.success("链路日志查询成功", data);
        } catch (Exception e) {
            log.error("链路日志查询失败", e);
            return ToolResult.error("链路日志查询失败: " + e.getMessage());
        }
    }

    /**
     * 查询指定服务的最新日志
     */
    // 查询某个微服务的最新日志，用于快速看“这个服务最近在发生什么”。
    @Tool(description = "查询指定微服务的最新日志记录")
    public ToolResult getLatestLogs(
            @ToolParam(description = "服务名称，如：gateway-service、order-service、user-service等") String serviceName,
            @ToolParam(description = "日志级别，可选。如：INFO、WARN、ERROR、DEBUG", required = false) String level,
            @ToolParam(description = "返回的日志条数，默认20条", required = false) Integer size) {

        try {
            // 结果条数保护。
            int limit = (size != null && size > 0) ? Math.min(size, 100) : 20;

            // 先按服务名过滤。
            LambdaEsQueryWrapper<LogDocument> wrapper = new LambdaEsQueryWrapper<>();
            wrapper.match(LogDocument::getProjectName, serviceName);

            // 如果指定了日志级别，再做二次过滤。
            if (level != null && !level.isEmpty()) {
                wrapper.match(LogDocument::getLevel, level.toUpperCase());
            }

            // 最新日志按时间倒序返回。
            wrapper.orderByDesc(LogDocument::getTimestamp);
            wrapper.limit(limit);

            List<LogDocument> logs = logMapper.selectList(wrapper);

            // 返回结构中除了日志列表，还会带上服务名、级别和条数等摘要信息。
            Map<String, Object> data = new HashMap<>();
            data.put("服务名称", serviceName);
            data.put("日志级别", level != null ? level : "全部");
            data.put("日志数量", logs.size());
            data.put("日志列表", formatLogs(logs));

            return ToolResult.success("日志查询成功", data);
        } catch (Exception e) {
            log.error("日志查询失败", e);
            return ToolResult.error("日志查询失败: " + e.getMessage());
        }
    }

    /**
     * 查询错误日志
     */
    // 错误日志查询工具：适合做问题初筛，先看最近有哪些 ERROR。
    @Tool(description = "查询系统中的错误日志（ERROR级别），可指定服务和时间范围")
    public ToolResult getErrorLogs(
            @ToolParam(description = "服务名称，可选。不填则查询所有服务的错误日志", required = false) String serviceName,
            @ToolParam(description = "返回的日志条数，默认30条", required = false) Integer size) {

        try {
            int limit = (size != null && size > 0) ? Math.min(size, 100) : 30;

            // 固定筛选 ERROR 级别。
            LambdaEsQueryWrapper<LogDocument> wrapper = new LambdaEsQueryWrapper<>();
            wrapper.match(LogDocument::getLevel, "ERROR");

            // 如果指定服务，就只看该服务的错误。
            if (serviceName != null && !serviceName.isEmpty()) {
                wrapper.match(LogDocument::getProjectName, serviceName);
            }

            wrapper.orderByDesc(LogDocument::getTimestamp);
            wrapper.limit(limit);

            List<LogDocument> logs = logMapper.selectList(wrapper);

            // 顺便按服务聚合统计错误数，方便模型快速判断哪一个服务错误最集中。
            Map<String, Long> errorCountByService = logs.stream()
                    .collect(Collectors.groupingBy(LogDocument::getProjectName, Collectors.counting()));

            Map<String, Object> data = new HashMap<>();
            data.put("查询范围", serviceName != null ? serviceName : "全部服务");
            data.put("错误日志数量", logs.size());
            data.put("各服务错误数", errorCountByService);
            data.put("错误日志列表", formatLogs(logs));

            return ToolResult.success("错误日志查询成功", data);
        } catch (Exception e) {
            log.error("错误日志查询失败", e);
            return ToolResult.error("错误日志查询失败: " + e.getMessage());
        }
    }

    /**
     * 查询警告日志
     */
    // WARN 查询和 ERROR 查询类似，但适合做风险预警和趋势观察。
    @Tool(description = "查询系统中的警告日志（WARN级别）")
    public ToolResult getWarnLogs(
            @ToolParam(description = "服务名称，可选。不填则查询所有服务的警告日志", required = false) String serviceName,
            @ToolParam(description = "返回的日志条数，默认30条", required = false) Integer size) {

        try {
            int limit = (size != null && size > 0) ? Math.min(size, 100) : 30;

            LambdaEsQueryWrapper<LogDocument> wrapper = new LambdaEsQueryWrapper<>();
            wrapper.match(LogDocument::getLevel, "WARN");

            if (serviceName != null && !serviceName.isEmpty()) {
                wrapper.match(LogDocument::getProjectName, serviceName);
            }

            wrapper.orderByDesc(LogDocument::getTimestamp);
            wrapper.limit(limit);

            List<LogDocument> logs = logMapper.selectList(wrapper);

            Map<String, Object> data = new HashMap<>();
            data.put("查询范围", serviceName != null ? serviceName : "全部服务");
            data.put("警告日志数量", logs.size());
            data.put("警告日志列表", formatLogs(logs));

            return ToolResult.success("警告日志查询成功", data);
        } catch (Exception e) {
            log.error("警告日志查询失败", e);
            return ToolResult.error("警告日志查询失败: " + e.getMessage());
        }
    }

    /**
     * 日志统计概览
     */
    // 日志统计概览工具：输出各微服务在 ERROR/WARN/INFO/DEBUG 四个级别上的分布情况。
    @Tool(description = "获取各微服务的日志统计概览，包括各级别日志的数量分布")
    public ToolResult getLogStatistics(
            @ToolParam(description = "服务名称，可选。不填则统计所有服务", required = false) String serviceName) {

        try {
            // 固定统计四种常见日志级别。
            List<String> levels = Arrays.asList("ERROR", "WARN", "INFO", "DEBUG");
            
            // 先确定统计哪些服务：如果用户指定了服务，就只统计一个；否则统计所有服务。
            List<String> services;
            if (serviceName != null && !serviceName.isEmpty()) {
                services = Collections.singletonList(serviceName);
            } else {
                // 使用 ES 聚合动态获取所有服务列表。
                services = getServiceListFromEs();
            }

            // 结果结构：service -> {ERROR: xx, WARN: xx, INFO: xx, DEBUG: xx}
            Map<String, Map<String, Long>> statistics = new LinkedHashMap<>();

            // 对每个服务分别统计四种日志级别数量。
            for (String service : services) {
                Map<String, Long> levelCounts = new LinkedHashMap<>();
                for (String level : levels) {
                    LambdaEsQueryWrapper<LogDocument> wrapper = new LambdaEsQueryWrapper<>();
                    wrapper.match(LogDocument::getProjectName, service);
                    wrapper.match(LogDocument::getLevel, level);
                    Long count = logMapper.selectCount(wrapper);
                    levelCounts.put(level, count);
                }
                statistics.put(service, levelCounts);
            }

            // 最终返回统计范围和统计明细。
            Map<String, Object> data = new HashMap<>();
            data.put("统计范围", serviceName != null ? serviceName : "全部服务");
            data.put("日志统计", statistics);

            return ToolResult.success("日志统计成功", data);
        } catch (Exception e) {
            log.error("日志统计失败", e);
            return ToolResult.error("日志统计失败: " + e.getMessage());
        }
    }

    /**
     * 按类名或方法名搜索日志
     */
    // 代码定位型查询：当用户说“查一下 UserController 的日志”时，可以用这个工具快速缩小范围。
    @Tool(description = "根据类名或方法名搜索日志，用于定位特定代码位置的日志")
    public ToolResult searchLogsByClass(
            @ToolParam(description = "类名，支持模糊匹配，如：UserController、OrderService") String className,
            @ToolParam(description = "方法名，可选", required = false) String methodName,
            @ToolParam(description = "返回的日志条数，默认20条", required = false) Integer size) {

        try {
            int limit = (size != null && size > 0) ? Math.min(size, 100) : 20;

            LambdaEsQueryWrapper<LogDocument> wrapper = new LambdaEsQueryWrapper<>();
            wrapper.match(LogDocument::getSourceClass, className);

            if (methodName != null && !methodName.isEmpty()) {
                wrapper.match(LogDocument::getSourceMethod, methodName);
            }

            wrapper.orderByDesc(LogDocument::getTimestamp);
            wrapper.limit(limit);

            List<LogDocument> logs = logMapper.selectList(wrapper);

            Map<String, Object> data = new HashMap<>();
            data.put("查询类名", className);
            data.put("查询方法名", methodName != null ? methodName : "不限");
            data.put("日志数量", logs.size());
            data.put("日志列表", formatLogs(logs));

            return ToolResult.success("日志搜索成功", data);
        } catch (Exception e) {
            log.error("日志搜索失败", e);
            return ToolResult.error("日志搜索失败: " + e.getMessage());
        }
    }

    /**
     * 格式化日志列表
     */
    private List<Map<String, Object>> formatLogs(List<LogDocument> logs) {
        // 统一把日志实体转换成更适合模型阅读的结构化 map 列表。
        return logs.stream().map(this::formatLog).collect(Collectors.toList());
    }

    /**
     * 格式化单条日志
     */
    private Map<String, Object> formatLog(LogDocument log) {
        // 输出字段尽量用中文 key，便于模型直接理解并在回答中组织内容。
        Map<String, Object> formatted = new LinkedHashMap<>();
        formatted.put("时间", log.getTimestamp());
        formatted.put("服务", log.getProjectName());
        formatted.put("级别", log.getLevel());
        formatted.put("消息", log.getMessage());
        formatted.put("traceId", log.getTraceId());
        formatted.put("类", log.getSourceClass());
        formatted.put("方法", log.getSourceMethod());
        formatted.put("行号", log.getSourceLine());
        formatted.put("线程", log.getThread());
        return formatted;
    }

    /**
     * 构建查询描述
     */
    private String buildQueryDesc(String keyword, String serviceName, String level) {
        // 把查询条件拼成一段可读文本，方便模型在总结结果时带出“本次查了什么”。
        StringBuilder sb = new StringBuilder();
        sb.append("关键词=").append(keyword);
        if (serviceName != null && !serviceName.isEmpty()) {
            sb.append(", 服务=").append(serviceName);
        }
        if (level != null && !level.isEmpty()) {
            sb.append(", 级别=").append(level);
        }
        return sb.toString();
    }

    /**
     * 工具返回结果的包装类
     */
    @Data
    public static class ToolResult {
        // success 表示工具调用是否成功。
        private boolean success;
        // message 用于给模型或调用方一个简短结论。
        private String message;
        // data 存放真正的结构化结果数据。
        private Object data;

        public static ToolResult success(String message, Object data) {
            // 成功结果工厂方法，统一构造返回体。
            ToolResult result = new ToolResult();
            result.setSuccess(true);
            result.setMessage(message);
            result.setData(data);
            return result;
        }

        public static ToolResult error(String message) {
            // 失败结果工厂方法，统一构造错误返回体。
            ToolResult result = new ToolResult();
            result.setSuccess(false);
            result.setMessage(message);
            return result;
        }
    }
}
