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


@AllArgsConstructor
@Slf4j
public class MarkdownLoader {

    // Spring 资源解析器，用来扫描 classpath 下的 markdown 规则文件。
    private final ResourcePatternResolver resourcePatternResolver;
    
    // 加载规则知识库中的全部 markdown 文档，并转换成可用于向量化检索的 Document 列表。
    public List<Document> loadMarkdowns() {
        List<Document> allDocuments = new ArrayList<>();
        try {
            // 扫描 classpath:datum 目录下的所有 .md 规则文档。
            Resource[] resources = resourcePatternResolver.getResources("classpath:datum/*.md");
            log.info("找到 {} 个Markdown文件", resources.length);
            for (Resource resource : resources) {
                // 当前处理的文件名，后续会作为文档标签和元数据来源。
                String fileName = resource.getFilename();
                log.info("正在处理文件: {}", fileName);
                
                // 默认标签先用文件名；如果文件名符合“标签-名称.md”格式，则取前半部分作为标签。
                String label = fileName;
                if (StringUtil.isNotEmpty(fileName)) {
                    final String[] parts = fileName.split("-");
                    if (parts.length > 1) {
                        label = parts[0];
                    }
                }
                log.info("提取的文档标签: {}", label);
  
                // 构建 markdown 读取配置。
                // 这里按水平分割线拆文档，并且忽略代码块、引用块，避免噪声内容进入规则知识库。
                Builder builder = MarkdownDocumentReaderConfig.builder()
                        // 按水平分割线分块
                        .withHorizontalRuleCreateDocument(true)
                        .withIncludeCodeBlock(false)
                        .withIncludeBlockquote(false);
                // 给文档片段附加原始文件名，便于检索后回溯来源。
                if (StringUtil.isNotEmpty(fileName)) {
                    builder.withAdditionalMetadata("name", fileName);
                }
                // 添加业务标签，比如退票、订票等，便于后续做过滤或调试。
                if (StringUtil.isNotEmpty(label)) {
                    builder.withAdditionalMetadata("label", label);
                }
                // 根据文件名提取关键词同义词，提升检索召回能力。
                String keywords = extractKeywords(fileName);
                if (StringUtil.isNotEmpty(keywords)) {
                    builder.withAdditionalMetadata("keywords", keywords);
                }
                // 标记数据来源和加载时间，便于后续排查知识库版本问题。
                builder.withAdditionalMetadata("source", "official_faq");
                builder.withAdditionalMetadata("loadTime", LocalDateTime.now().toString());
                MarkdownDocumentReaderConfig config = builder.build(); 
                // 用配置好的 reader 将 markdown 文件解析成一个或多个文档片段。
                MarkdownDocumentReader markdownDocumentReader = new MarkdownDocumentReader(resource, config);
                List<Document> documents = markdownDocumentReader.get();
                log.info("文件 {} 加载了 {} 个文档片段", fileName, documents.size());
                // 先收集原始切分后的文档片段。
                allDocuments.addAll(documents);
            }
            log.info("总共加载了 {} 个文档片段", allDocuments.size());
            List<Document> splitDocuments = new ArrayList<>();
            // 对过长文档做二次 token 级切分，避免单片段过大影响向量化和召回精度。
            TokenTextSplitter splitter = new TokenTextSplitter(400, 50, 5, 10000, true);
            
            for (Document doc : allDocuments) {
                // 只有文本过长的片段才进行二次切分，短文本直接保留即可。
                if (doc.getText() != null && doc.getText().length() > 1000) {
                    List<Document> splits = splitter.split(List.of(doc));
                    log.info("文档[{}]过长，切分为{}个片段",
                            doc.getMetadata().get("name"), splits.size());
                    splitDocuments.addAll(splits);
                } else {
                    splitDocuments.add(doc);
                }
            }
            log.info("二次切分后总共 {} 个文档片段", splitDocuments.size());
            // 返回最终可以写入向量库/混合检索缓存的文档片段列表。
            return splitDocuments;
        } catch (IOException e) {
           // 文档加载失败时记录异常，避免应用静默失败。
           log.error("Markdown 文档加载失败", e);
        }
        // 异常情况下返回已成功加载的部分文档，尽量减少系统完全不可用的概率。
        return allDocuments;
    }
    
    // 根据文件名补充关键词元数据。
    // 这样像“退票”“退款”“取消订单”这类同义表达在检索时更容易召回同一份规则文档。
    private String extractKeywords(String fileName) {
        if (StringUtil.isEmpty(fileName)) {
            return "";
        }
        // 这里维护的是一个简单的关键词映射表，可按业务需要继续扩展。
        Map<String, String> keywordMap = Map.of(
            "退票", "退票,退款,取消订单,退钱",
            "订票", "订票,购票,买票,下单",
            "取消", "取消,作废,退款"
        );
        
        StringBuilder keywords = new StringBuilder();
        // 如果文件名里包含某个业务关键词，就把对应同义词都写入 metadata 中。
        for (Map.Entry<String, String> entry : keywordMap.entrySet()) {
            if (fileName.contains(entry.getKey())) {
                if (keywords.length() > 0) {
                   keywords.append(",");
                }
                keywords.append(entry.getValue());
            }
        }
        // 返回最终提取出的关键词字符串。
        return keywords.toString();
    }
}
