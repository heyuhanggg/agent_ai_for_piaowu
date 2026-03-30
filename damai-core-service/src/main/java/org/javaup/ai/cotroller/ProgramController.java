package org.javaup.ai.cotroller;

import jakarta.annotation.Resource;
import org.javaup.ai.ai.function.call.ProgramCall;
import org.javaup.ai.ai.function.dto.ProgramSearchFunctionDto;
import org.javaup.ai.dto.ProgramDetailDto;
import org.javaup.ai.service.HybridSearchService;
import org.javaup.ai.vo.ProgramSearchVo;
import org.javaup.ai.vo.result.ProgramDetailResultVo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.stream.Collectors;

import static org.javaup.ai.constants.DaMaiConstant.RAG_VERSION;


@RestController
@RequestMapping("/program")
public class ProgramController {

    private static final Logger log = LoggerFactory.getLogger(ProgramController.class);

    // 节目基础查询能力，给搜索和详情接口复用。
    @Autowired
    private ProgramCall programCall;

    // 贴心助手对话客户端。
    @Resource
    private ChatClient assistantChatClient;
    
    // 规则助手对话客户端，对应 markdown / RAG 能力。
    @Resource
    private ChatClient markdownChatClient;
    
    // 运维助手对话客户端，对应分析和 MCP 工具调用场景。
    @Resource
    private ChatClient analysisChatClient;
    
    // 混合检索服务：规则助手 V2 中用于关键词 + 向量的组合召回。
    @Resource
    private HybridSearchService hybridSearchService;
    
    // 当前启用的 RAG 版本号，1 表示 QuestionAnswerAdvisor 版，2 表示混合检索增强版。
    @Value("${"+RAG_VERSION+":1}")
    private Integer ragVersion;
    
    // 普通贴心助手聊天接口。
    // 这里不走规则知识库，只是把用户问题直接送给 assistantChatClient。
    @RequestMapping(value = "/chat", produces = "text/html;charset=utf-8")
    public Flux<String> chat(@RequestParam("prompt") String prompt,
                                @RequestParam("chatId") String chatId) {
        // 把用户问题发送给贴心助手模型，并带上 chatId 作为会话记忆标识。
        return assistantChatClient.prompt()
                .user(prompt)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, chatId))
                .stream()
                .content();
    }
    
    // 规则助手接口。
    // 这是“演出票务规则助手”的核心入口，负责基于知识库回答退票、订票、规则类问题。
    @RequestMapping(value = "/rag", produces = "text/html;charset=utf-8")
    public Flux<String> rag(@RequestParam("prompt") String prompt,
                             @RequestParam("chatId") String chatId) {
        final Integer ragTwoVersionValue = 2;
        // 如果当前启用的是 V2 版本，则先走混合检索，再手工把上下文拼到 prompt 里。
        if (ragVersion.equals(ragTwoVersionValue)) {
            // 从混合检索服务中拿到与当前问题最相关的规则文档片段。
            List<Document> documents = hybridSearchService.hybridSearch(prompt, 10, true);
            log.info("混合检索返回 {} 个文档", documents.size());
            
            // 把检索结果文本拼接成一段上下文，后面统一交给模型参考作答。
            String context = documents.stream()
                    .map(Document::getText)
                    .collect(Collectors.joining("\n\n"));
            
            // 手工构造增强版提示词：明确告诉模型“必须优先基于检索上下文回答”。
            String enhancedPrompt = """
                以下是检索到的相关上下文信息：
                ---------------------
                %s
                ---------------------
                请基于上述上下文信息回答用户问题。如果上下文中没有相关信息，请告知用户。
                用户问题：%s
                """.formatted(context, prompt);
            
            // 将增强后的 prompt 发给规则助手模型，同时保留会话记忆。
            return markdownChatClient.prompt()
                    .user(enhancedPrompt)
                    .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, chatId))
                    .stream()
                    .content();
        }
        // 如果是 V1 版本，则直接把原始问题交给 markdownChatClient。
        // 检索和上下文拼接会由 ChatClient 内部挂载的 QuestionAnswerAdvisor 自动完成。
        return markdownChatClient.prompt()
                .user(prompt)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, chatId))
                .stream()
                .content();
    }
    
    // 运维助手 MCP 接口。
    // 这个方法不是规则助手主链路，但放在同一个控制器里，供前端调用带工具能力的分析助手。
    @RequestMapping(value = "/chat/mcp", produces = "text/html;charset=utf-8")
    public Flux<String> chatMcp(@RequestParam("prompt") String prompt,
                             @RequestParam("chatId") String chatId) {
        // MCP 工具已在 analysisChatClient 中全局配置，这里直接发送用户问题即可。
        return analysisChatClient.prompt()
                .user(prompt)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, chatId))
                .stream()
                .content();
    }
    
    // 节目搜索接口：给前端或其他服务提供节目检索能力。
    @PostMapping(value = "/search")
    public List<ProgramSearchVo> search(@RequestBody ProgramSearchFunctionDto programSearchFunctionDto) {
        return programCall.search(programSearchFunctionDto);
    }

    // 节目详情接口：根据节目 id 查询完整详情。
    @PostMapping(value = "/detail")
    public ProgramDetailResultVo search(@RequestBody ProgramDetailDto programDetailDto) {
        return programCall.detail(programDetailDto);
    }
}
