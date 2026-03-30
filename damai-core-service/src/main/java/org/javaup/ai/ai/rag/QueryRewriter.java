package org.javaup.ai.ai.rag;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.deepseek.DeepSeekChatModel;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.preretrieval.query.transformation.QueryTransformer;
import org.springframework.ai.rag.preretrieval.query.transformation.RewriteQueryTransformer;
import org.springframework.stereotype.Component;


@Component
public class QueryRewriter {

    // 查询转换器：负责把用户原始问题重写成更适合检索系统理解的查询文本。
    private final QueryTransformer queryTransformer;

    public QueryRewriter(DeepSeekChatModel model) {
        // 定义查询改写提示词。
        // 这里明确限制查询重写只围绕“演出/演唱会退票规则”，避免召回到机票、火车票等无关票务内容。
        PromptTemplate template = PromptTemplate.builder().template("""
                你是演出/演唱会退票规则专家。
                请将用户查询重写为最适合用在 {target} 中检索演出门票退票规则的版本。

                用户原始查询："{query}"

                注意：不要提及机票、火车票等交通票务，只专注演唱会/演出门票退票规则。
                """).build();
        // 创建查询重写转换器。
        // 它的作用是在正式检索前先优化 query 表达，提高规则类知识库的召回率和准确率。
        queryTransformer = RewriteQueryTransformer.builder()
                .chatClientBuilder(ChatClient.builder(model))
                .promptTemplate(template)
                .targetSearchSystem("节目和演唱会规则向量库")
                .build();
    }


    public String doQueryRewrite(String prompt) {
        // 执行查询重写，把用户原问题转换成更适合检索的文本。
        return queryTransformer.transform(new Query(prompt)).text();
    }
}
