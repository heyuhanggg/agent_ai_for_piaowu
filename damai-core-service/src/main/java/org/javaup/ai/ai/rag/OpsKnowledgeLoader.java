package org.javaup.ai.ai.rag;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.javaup.ai.utils.StringUtil;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.markdown.MarkdownDocumentReader;
import org.springframework.ai.reader.markdown.config.MarkdownDocumentReaderConfig;
import org.springframework.ai.reader.markdown.config.MarkdownDocumentReaderConfig.Builder;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 运维知识库文档加载器
 * 
 * 负责加载 classpath:ops-knowledge/ 目录下的运维手册、SOP、故障案例等 Markdown 文档，
 * 转换为可用于向量化检索的 Document 列表。
 */
@AllArgsConstructor
@Slf4j
public class OpsKnowledgeLoader {

    private final ResourcePatternResolver resourcePatternResolver;

    /**
     * 加载运维知识库中的全部 Markdown 文档
     */
    public List<Document> loadOpsKnowledge() {
        List<Document> allDocuments = new ArrayList<>();
        try {
            Resource[] resources = resourcePatternResolver.getResources("classpath:ops-knowledge/*.md");
            log.info("运维知识库：找到 {} 个Markdown文件", resources.length);

            for (Resource resource : resources) {
                String fileName = resource.getFilename();
                log.info("运维知识库：正在处理文件: {}", fileName);

                // 提取文档类别标签
                String category = extractCategory(fileName);
                log.info("运维知识库：文档类别: {}", category);

                Builder builder = MarkdownDocumentReaderConfig.builder()
                        // 按水平分割线分块，每个故障/SOP/案例独立成一个文档片段
                        .withHorizontalRuleCreateDocument(true)
                        .withIncludeCodeBlock(true)   // 运维文档中的代码块有价值（配置、命令等）
                        .withIncludeBlockquote(true);  // 运维文档中的引用块可能包含重要提示

                if (StringUtil.isNotEmpty(fileName)) {
                    builder.withAdditionalMetadata("name", fileName);
                }
                if (StringUtil.isNotEmpty(category)) {
                    builder.withAdditionalMetadata("category", category);
                }

                // 提取运维领域关键词
                String keywords = extractOpsKeywords(fileName);
                if (StringUtil.isNotEmpty(keywords)) {
                    builder.withAdditionalMetadata("keywords", keywords);
                }

                builder.withAdditionalMetadata("source", "ops_knowledge_base");
                builder.withAdditionalMetadata("loadTime", LocalDateTime.now().toString());

                MarkdownDocumentReaderConfig config = builder.build();
                MarkdownDocumentReader reader = new MarkdownDocumentReader(resource, config);
                List<Document> documents = reader.get();
                log.info("运维知识库：文件 {} 加载了 {} 个文档片段", fileName, documents.size());
                allDocuments.addAll(documents);
            }

            log.info("运维知识库：总共加载了 {} 个文档片段", allDocuments.size());

            // 对过长文档做二次 token 级切分
            List<Document> splitDocuments = new ArrayList<>();
            TokenTextSplitter splitter = new TokenTextSplitter(500, 80, 5, 10000, true);

            for (Document doc : allDocuments) {
                if (doc.getText() != null && doc.getText().length() > 1200) {
                    List<Document> splits = splitter.split(List.of(doc));
                    log.info("运维知识库：文档[{}]过长({}字符)，切分为{}个片段",
                            doc.getMetadata().get("name"), doc.getText().length(), splits.size());
                    splitDocuments.addAll(splits);
                } else {
                    splitDocuments.add(doc);
                }
            }

            log.info("运维知识库：二次切分后总共 {} 个文档片段", splitDocuments.size());
            return splitDocuments;
        } catch (IOException e) {
            log.error("运维知识库：Markdown 文档加载失败", e);
        }
        return allDocuments;
    }

    /**
     * 根据文件名提取文档类别
     */
    private String extractCategory(String fileName) {
        if (StringUtil.isEmpty(fileName)) {
            return "general";
        }
        if (fileName.contains("故障排查") || fileName.contains("排查手册")) {
            return "troubleshooting";
        }
        if (fileName.contains("SOP") || fileName.contains("操作流程")) {
            return "sop";
        }
        if (fileName.contains("案例") || fileName.contains("复盘")) {
            return "case_study";
        }
        if (fileName.contains("架构") || fileName.contains("依赖")) {
            return "architecture";
        }
        return "general";
    }

    /**
     * 提取运维领域关键词及同义词
     */
    private String extractOpsKeywords(String fileName) {
        if (StringUtil.isEmpty(fileName)) {
            return "";
        }

        Map<String, String> keywordMap = Map.ofEntries(
                Map.entry("故障", "故障,异常,报错,错误,error,exception,问题"),
                Map.entry("排查", "排查,诊断,分析,定位,排障,troubleshoot"),
                Map.entry("SOP", "SOP,标准操作,流程,步骤,操作手册"),
                Map.entry("OOM", "OOM,内存溢出,OutOfMemoryError,堆内存,heap"),
                Map.entry("超时", "超时,timeout,响应慢,延迟,latency"),
                Map.entry("数据库", "数据库,MySQL,SQL,慢查询,连接池,database"),
                Map.entry("Redis", "Redis,缓存,缓存雪崩,缓存穿透,缓存击穿"),
                Map.entry("发布", "发布,部署,回滚,上线,deploy,rollback"),
                Map.entry("扩容", "扩容,缩容,弹性伸缩,scale"),
                Map.entry("案例", "案例,复盘,事故,故障回顾"),
                Map.entry("架构", "架构,拓扑,依赖,微服务,服务调用"),
                Map.entry("监控", "监控,告警,指标,metrics,alert"),
                Map.entry("日志", "日志,log,错误日志,日志查询"),
                Map.entry("链路", "链路,trace,链路追踪,调用链")
        );

        StringBuilder keywords = new StringBuilder();
        for (Map.Entry<String, String> entry : keywordMap.entrySet()) {
            if (fileName.contains(entry.getKey())) {
                if (keywords.length() > 0) {
                    keywords.append(",");
                }
                keywords.append(entry.getValue());
            }
        }
        return keywords.toString();
    }
}
