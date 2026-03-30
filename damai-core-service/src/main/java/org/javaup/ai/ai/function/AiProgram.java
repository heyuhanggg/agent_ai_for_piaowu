package org.javaup.ai.ai.function;

import cn.hutool.core.collection.CollectionUtil;
import org.javaup.ai.ai.function.dto.CreateOrderFunctionDto;
import org.javaup.ai.ai.function.dto.ProgramRecommendFunctionDto;
import org.javaup.ai.ai.function.dto.ProgramSearchFunctionDto;
import org.javaup.ai.dto.ProgramDetailDto;
import org.javaup.ai.dto.ProgramOrderCreateDto;
import org.javaup.ai.dto.TicketCategoryListByProgramDto;
import org.javaup.ai.ai.function.call.OrderCall;
import org.javaup.ai.ai.function.call.ProgramCall;
import org.javaup.ai.ai.function.call.TicketCategoryCall;
import org.javaup.ai.ai.function.call.UserCall;
import org.javaup.ai.utils.StringUtil;
import org.javaup.ai.vo.CreateOrderVo;
import org.javaup.ai.vo.ProgramDetailVo;
import org.javaup.ai.vo.ProgramSearchVo;
import org.javaup.ai.vo.TicketCategoryDetailVo;
import org.javaup.ai.vo.TicketCategoryVo;
import org.javaup.ai.vo.TicketUserVo;
import org.javaup.ai.vo.UserDetailVo;
import org.javaup.ai.vo.result.ProgramDetailResultVo;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.javaup.ai.constants.DaMaiConstant.ORDER_LIST_ADDRESS;


@Component
public class AiProgram {

    // 节目能力封装，负责节目推荐、节目搜索和节目详情查询。
    @Autowired
    private ProgramCall programCall;

    // 票档能力封装，负责根据节目 id 查询对应票档及库存信息。
    @Autowired
    private TicketCategoryCall ticketCategoryCall;
    
    // 用户能力封装，负责根据手机号查询用户、根据用户 id 查询购票人。
    @Autowired
    private UserCall userCall;
    
    // 订单能力封装，负责调用后端真实下单接口生成订单。
    @Autowired
    private OrderCall orderCall;
    
    // 推荐节目工具：给大模型一个“按地区/类型找节目推荐”的能力入口。
    // 这里不自己做业务判断，直接把筛选条件转给 ProgramCall 查询 ES 中的节目数据。
    @Tool(description = "根据地区或者类型查询推荐的节目")
    public List<ProgramSearchVo> selectProgramRecommendList(@ToolParam(description = "查询的条件", required = true) ProgramRecommendFunctionDto programRecommendFunctionDto){
        return programCall.recommendList(programRecommendFunctionDto);
    }

    // 节目搜索工具：按城市、艺人、演出时间等条件查节目列表。
    // 这里返回的是列表结果，适合先让模型找到候选节目。
    @Tool(description = "根据条件查询节目")
    public List<ProgramSearchVo> selectProgramList(@ToolParam(description = "查询的条件", required = true) ProgramSearchFunctionDto programSearchFunctionDto){
        return programCall.search(programSearchFunctionDto);
    }
    
    // 节目详情工具：当前实现直接复用 selectTicketCategory。
    // 也就是说，查详情时会顺便把票档和库存一起补齐，避免模型拿到一个“只有节目没有票档”的半成品数据。
    @Tool(description = "根据条件查询节目和演唱会的详情")
    public ProgramDetailVo detail(@ToolParam(description = "查询的条件", required = true) ProgramSearchFunctionDto programSearchFunctionDto){
        return selectTicketCategory(programSearchFunctionDto);
    }

    // 票档查询工具：这是购票链路里的关键聚合方法。
    // 它的职责不是只查票档，而是先定位节目，再查节目详情，最后再把票档库存信息合并到节目详情对象中返回。
    @Tool(description = "根据条件查询节目和演唱会的票档信息")
    public ProgramDetailVo selectTicketCategory(@ToolParam(description = "查询的条件", required = true) ProgramSearchFunctionDto programSearchFunctionDto){
        // 第一步：根据用户给的城市、艺人、时间等条件先搜索节目。
        List<ProgramSearchVo> programSearchVoList = programCall.search(programSearchFunctionDto);
        // 如果连节目都查不到，说明当前条件无法定位到目标演出，直接返回空。
        if (CollectionUtil.isEmpty(programSearchVoList)) {
            return null;
        }
        // 当前实现只取第一条节目作为后续详情和票档查询目标。
        // 这意味着如果有多个候选节目，默认以搜索结果的第一条为准。
        ProgramSearchVo programSearchVo = programSearchVoList.get(0);
        // 组装节目详情接口需要的 DTO，只传节目 id 给后端详情接口。
        ProgramDetailDto programDetailDto = new ProgramDetailDto();
        programDetailDto.setId(programSearchVo.getId());
        // 查询节目详情，拿到更完整的节目基础信息和原始票档列表。
        ProgramDetailResultVo programDetailResultVo = programCall.detail(programDetailDto);
        // 如果详情接口没有返回 data，说明节目详情不可用，直接返回空。
        if (Objects.isNull(programDetailResultVo.getData())) {
            return null;
        }
        // 取出详情数据，后续要在这个对象基础上补票档库存。
        ProgramDetailVo programDetailVo = programDetailResultVo.getData();
        // 组装票档查询参数，核心是节目 id。
        TicketCategoryListByProgramDto ticketCategoryListByProgramDto = new TicketCategoryListByProgramDto();
        ticketCategoryListByProgramDto.setProgramId(programDetailVo.getId());
        // 查询该节目下所有票档的明细信息，比如库存、总票数等。
        List<TicketCategoryDetailVo> ticketCategoryDetailVoList = ticketCategoryCall.selectListByProgram(ticketCategoryListByProgramDto);
        // 把票档明细转成 map，key 是票档 id，方便后面 O(1) 地回填到节目详情里的票档列表。
        Map<Long, TicketCategoryDetailVo> ticketCategoryDetailMap = ticketCategoryDetailVoList.stream()
                .collect(Collectors.toMap(TicketCategoryDetailVo::getId,
                        ticketCategoryDetailVo -> ticketCategoryDetailVo,
                        (v1, v2) -> v2));
        // 遍历节目详情里的票档列表，把查询到的剩余库存和总库存补进去。
        for (TicketCategoryVo ticketCategoryVo : programDetailVo.getTicketCategoryVoList()) {
            TicketCategoryDetailVo ticketCategoryDetailVo = ticketCategoryDetailMap.get(ticketCategoryVo.getId());
            // 只有当前票档 id 在明细里存在时才进行库存回填，避免空指针。
            if (Objects.nonNull(ticketCategoryDetailVo)) {
                ticketCategoryVo.setRemainNumber(ticketCategoryDetailVo.getRemainNumber());
                ticketCategoryVo.setTotalNumber(ticketCategoryDetailVo.getTotalNumber());
            }
        }
        // 返回的不是单纯票档列表，而是“节目详情 + 已补齐库存的票档信息”的聚合结果。
        return programDetailVo;
    }
    
    // 下单工具：这是购票助手里最核心的编排逻辑。
    // 模型不会直接传内部的 programId/userId/ticketCategoryId，而是先传用户看得懂的条件，服务端再做真实映射和校验。
    @Tool(description = "生成用户购买节目的订单，返回订单号")
    public CreateOrderVo createOrder(@ToolParam(description = "查询的条件", required = true) CreateOrderFunctionDto createOrderFunctionDto){
        // 先把下单 DTO 中与节目搜索相关的字段拷贝出来，作为查节目和票档的条件。
        ProgramSearchFunctionDto programSearchFunctionDto = new ProgramSearchFunctionDto();
        BeanUtils.copyProperties(createOrderFunctionDto, programSearchFunctionDto);
        // 先通过节目查询 + 票档聚合逻辑拿到真实节目和真实票档信息。
        ProgramDetailVo programDetailVo = selectTicketCategory(programSearchFunctionDto);
        // 节目不存在时直接拒绝下单，避免模型传了错误条件仍继续执行高风险操作。
        if (Objects.isNull(programDetailVo)) {
            throw new RuntimeException("没有查询到节目，请检查查询条件是否正确");
        }
        // 根据手机号查真实用户信息，手机号是 AI 工具层与后端用户体系的桥梁。
        UserDetailVo userDetailVo = userCall.userDetail(createOrderFunctionDto.getMobile());
        // 查不到用户则不能继续，因为后端订单一定要绑定真实 userId。
        if (Objects.isNull(userDetailVo)) {
            throw new RuntimeException("用户信息不存在");
        }
        // 根据用户 id 查询该用户名下全部购票人，为后续做购票人身份匹配准备数据。
        List<TicketUserVo> ticketUserVoList = userCall.ticketUserList(userDetailVo.getId());
        // 没有购票人信息则无法创建订单，因为订单必须明确买给谁。
        if (CollectionUtil.isEmpty(ticketUserVoList)) {
            throw new RuntimeException("购票人信息不存在");
        }
        // 用来保存最终匹配成功的购票人集合。
        List<TicketUserVo> ticketUserVoFilterList = new ArrayList<>();
        // 遍历当前用户名下的购票人，逐个和模型传来的证件号列表做匹配。
        for (final TicketUserVo ticketUserVo : ticketUserVoList) {
            for (final String number : createOrderFunctionDto.getTicketUserNumberList()) {
                // 出于隐私和实际展示场景考虑，这里不是拿完整证件号比较，而是比对前 4 位和后 4 位。
                String ticketUserNumberFirst = StringUtil.getFirstN(ticketUserVo.getIdNumber(),4);
                String ticketUserNumberLast = StringUtil.getLastN(ticketUserVo.getIdNumber(),4);
                
                String paramNumberFirst = StringUtil.getFirstN(number,4);
                String paramNumberLast = StringUtil.getLastN(number,4);
                
                // 当前后四位都匹配成功时，认为这个购票人与用户输入的证件号对应上了。
                if (ticketUserNumberFirst.equals(paramNumberFirst) && ticketUserNumberLast.equals(paramNumberLast)) {
                    ticketUserVoFilterList.add(ticketUserVo);
                }
            }
        }
        // 如果匹配到的购票人数和用户输入的人数不一致，说明有证件号错误或信息不完整，直接拒绝下单。
        if (ticketUserVoFilterList.size() != createOrderFunctionDto.getTicketUserNumberList().size()) {
            throw new RuntimeException("购票人信息不完整，请检查购票人信息是否正确");
        }
        // 这里不信任模型直接给内部票档 id，而是根据用户选择的票价，在真实票档列表里重新匹配票档 id。
        Long ticketCategoryId = null;
        for (final TicketCategoryVo ticketCategoryVo : programDetailVo.getTicketCategoryVoList()) {
            // 用价格匹配票档，匹配成功后拿到后端真实的票档 id。
            if (createOrderFunctionDto.getTicketCategoryPrice().compareTo(ticketCategoryVo.getPrice()) == 0) {
                ticketCategoryId = ticketCategoryVo.getId();
                break;
            }
        }
        // 如果根据真实节目票档都找不到对应价位，说明用户输入的票档信息无效。
        if (Objects.isNull(ticketCategoryId)) {
            throw new RuntimeException("没有查询到对应的票档信息");
        }
        // 开始组装真正提交给后端订单服务的 DTO。
        // 到这一步为止，内部需要的 programId/userId/ticketCategoryId 都已经通过服务端真实查询得到。
        ProgramOrderCreateDto programOrderCreateDto = new ProgramOrderCreateDto();
        programOrderCreateDto.setProgramId(programDetailVo.getId());
        programOrderCreateDto.setUserId(userDetailVo.getId());
        programOrderCreateDto.setTicketUserIdList(ticketUserVoFilterList.stream().map(TicketUserVo::getId).collect(Collectors.toList()));
        programOrderCreateDto.setTicketCategoryId(ticketCategoryId);
        programOrderCreateDto.setTicketCount(createOrderFunctionDto.getTicketCount());
        // 调用后端真实下单接口生成订单号。
        String orderNumber = orderCall.createOrder(programOrderCreateDto);
        // 把订单号和订单列表页地址封装给前端/调用方，便于后续跳转查看订单。
        CreateOrderVo createOrderVo = new CreateOrderVo();
        createOrderVo.setOrderNumber(orderNumber);
        createOrderVo.setOrderListAddress(ORDER_LIST_ADDRESS);
        return createOrderVo;
    }
}
