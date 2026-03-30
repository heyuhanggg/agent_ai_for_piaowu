package org.javaup.ai.ai.function.dto;

import lombok.Data;
import org.springframework.ai.tool.annotation.ToolParam;

import java.math.BigDecimal;
import java.util.Date;
import java.util.List;


@Data
public class CreateOrderFunctionDto {
    
    // 演出城市：用于先定位目标节目。
    // 模型不直接传 programId，而是先用城市等自然语言条件查节目。
    @ToolParam(required = true, description = "节目演出城市")
    private String cityName;
    
    // 艺人或节目主体：和城市一起用于确定用户要买哪一场节目。
    @ToolParam(required = true, description = "节目艺人或者节目明星")
    private String actor;
    
    // 演出时间：可选，用于缩小节目范围，避免同一个艺人多场演出时定位错节目。
    @ToolParam(required = false, description = "节目演出时间")
    private Date showTime;
    
    // 用户手机号：AI 层拿它去换取后端真实 userId。
    @ToolParam(required = true, description = "用户手机号")
    private String mobile;
    
    // 购票人证件号列表：AI 层不会直接传 ticketUserId，而是先传证件号，再由服务端去匹配真实购票人。
    @ToolParam(required = true, description = "购票人证件号码列表")
    private List<String> ticketUserNumberList;;
    
    // 票档价位：用户通常记得“看 580 的票”，所以这里用价格作为人类可理解的票档选择条件。
    // 服务端会再用这个价格去匹配真实 ticketCategoryId。
    @ToolParam(required = true, description = "节目的票档价位")
    private BigDecimal ticketCategoryPrice;
    
    // 购买数量：最终会透传给后端订单 DTO 的 ticketCount 字段。
    @ToolParam(required = true, description = "节目的票档购买数量")
    private Integer ticketCount;
}
