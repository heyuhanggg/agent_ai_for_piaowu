package org.javaup.ai.ai.function.call;


import cn.hutool.http.HttpRequest;
import com.alibaba.fastjson.JSON;
import org.dromara.easyes.core.conditions.select.LambdaEsQueryWrapper;
import org.dromara.easyes.core.kernel.EsWrappers;
import org.javaup.ai.ai.function.dto.ProgramRecommendFunctionDto;
import org.javaup.ai.ai.function.dto.ProgramSearchFunctionDto;
import org.javaup.ai.dto.ProgramDetailDto;
import org.javaup.ai.enums.BaseCode;
import org.javaup.ai.es.mapper.ProgramMapper;
import org.javaup.ai.utils.StringUtil;
import org.javaup.ai.vo.ProgramSearchVo;
import org.javaup.ai.vo.result.ProgramDetailResultVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;

import static org.javaup.ai.constants.DaMaiConstant.PROGRAM_DETAIL_URL;


@Component
public class ProgramCall {

    // ES 节目索引访问入口，用来按条件搜索节目列表。
    @Autowired
    private ProgramMapper programMapper;
    
    // 推荐节目查询：根据地区和节目分类筛选推荐结果。
    // 这里查的是 ES 中的节目索引，适合大模型先拿一批候选节目。
    public List<ProgramSearchVo> recommendList(ProgramRecommendFunctionDto programRecommendFunctionDto){
        // 按传入条件动态拼装 ES 查询条件；如果字段为空，则该条件不会参与过滤。
        LambdaEsQueryWrapper<ProgramSearchVo> wrapper = EsWrappers.lambdaQuery(ProgramSearchVo.class)
                .eq(StringUtil.isNotEmpty(programRecommendFunctionDto.getAreaName()), ProgramSearchVo::getAreaName, programRecommendFunctionDto.getAreaName())
                .eq(StringUtil.isNotEmpty(programRecommendFunctionDto.getProgramCategory()), ProgramSearchVo::getParentProgramCategoryName, programRecommendFunctionDto.getProgramCategory());
        // 执行 ES 查询并返回推荐节目列表。
        return programMapper.selectList(wrapper);
    }

    // 节目搜索：根据城市、艺人、演出时间查节目。
    // 这是购票助手最常用的“先定位节目”能力。
    public List<ProgramSearchVo> search(ProgramSearchFunctionDto programSearchFunctionDto){
        // 这里同样是动态条件拼装：有值才加筛选，避免强依赖所有字段都传。
        LambdaEsQueryWrapper<ProgramSearchVo> wrapper = EsWrappers.lambdaQuery(ProgramSearchVo.class)
                .eq(StringUtil.isNotEmpty(programSearchFunctionDto.getCityName()), ProgramSearchVo::getAreaName, programSearchFunctionDto.getCityName())
                .eq(StringUtil.isNotEmpty(programSearchFunctionDto.getActor()), ProgramSearchVo::getActor, programSearchFunctionDto.getActor())
                .ge(Objects.nonNull(programSearchFunctionDto.getShowTime()), ProgramSearchVo::getShowTime, programSearchFunctionDto.getShowTime());
        // 返回符合条件的节目列表，供上层进一步选节目、查详情或查票档。
        return programMapper.selectList(wrapper);
    }

    // 节目详情查询：这里不走 ES，而是直接调用后端节目详情接口拿完整数据。
    public ProgramDetailResultVo detail(ProgramDetailDto programDetailDto) {
        // 调用真实节目详情接口，把节目 id 发送给后端系统。
        String result = HttpRequest.post(PROGRAM_DETAIL_URL)
                .header("no_verify", "true")
                .body(JSON.toJSONString(programDetailDto))
                .timeout(20000)
                .execute().body();
        // 把接口返回的 JSON 反序列化成统一的结果对象。
        ProgramDetailResultVo programDetailResultVo = JSON.parseObject(result, ProgramDetailResultVo.class);
        // 如果返回码不是成功，直接抛异常，避免上层继续用错误数据执行后续流程。
        if (!Objects.equals(programDetailResultVo.getCode(), BaseCode.SUCCESS.getCode())) {
            throw new RuntimeException("调用演出票务系统查询节目失败");
        }
        // 成功时返回完整详情结果，data 里包含真正的节目详情信息。
        return programDetailResultVo;
    }
}
