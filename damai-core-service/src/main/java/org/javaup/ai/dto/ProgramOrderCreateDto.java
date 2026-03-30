package org.javaup.ai.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.media.Schema.RequiredMode;
import lombok.Data;

import java.util.List;


@Data
@Schema(title="ProgramOrderCreateDto", description ="节目订单创建")
public class ProgramOrderCreateDto {
    
    // 节目 id：后端真实节目主键，由 AI 工具层先查节目后再回填。
    @Schema(name ="programId", type ="Long", description ="节目id",requiredMode= RequiredMode.REQUIRED)
    private Long programId;
    
    // 用户 id：由手机号换取的后端真实用户主键。
    @Schema(name ="userId", type ="Long", description ="用户id",requiredMode= RequiredMode.REQUIRED)
    private Long userId;
    
    // 购票人 id 列表：由证件号匹配出当前用户名下真实购票人后得到。
    @Schema(name ="ticketUserIdList", type ="List<Long>", description ="购票人id集合",requiredMode= RequiredMode.REQUIRED)
    private List<Long> ticketUserIdList;
    
    // 节目票档 id：不是模型直接给的，而是服务端根据真实票档列表重新匹配出来的。
    @Schema(name ="ticketCategoryId", type ="Long", description = "节目票档id",requiredMode= RequiredMode.REQUIRED)
    private Long ticketCategoryId;
    
    // 购买张数：最终提交给后端订单服务的购票数量。
    @Schema(name ="ticketCount", type ="Integer", description = "购买票数量",requiredMode= RequiredMode.REQUIRED)
    private Integer ticketCount;
}
