package org.javaup.ai.ai.function.dto;

import lombok.Data;
import org.springframework.ai.tool.annotation.ToolParam;

import java.util.Date;


@Data
public class ProgramSearchFunctionDto {

    // 演出城市：用于把节目搜索范围限定到某个城市。
    // 例如用户说“帮我查北京的演唱会”，这里就会填“北京”。
    @ToolParam(required = false, description = "节目演出城市")
    private String cityName;

    // 艺人或节目主体：用于按明星、乐队、演员等核心主体筛选节目。
    // 这是定位目标节目的主要条件之一。
    @ToolParam(required = false, description = "节目艺人或者节目明星")
    private String actor;

    // 演出时间：用于过滤指定时间之后或指定场次附近的节目。
    // 当前搜索实现里会把它作为时间下界条件使用。
    @ToolParam(required = false, description = "节目演出时间")
    private Date showTime;
}
