package org.javaup.ai.ai.function.call;

import cn.hutool.core.collection.CollectionUtil;
import cn.hutool.http.HttpRequest;
import com.alibaba.fastjson.JSON;
import org.javaup.ai.dto.TicketCategoryListByProgramDto;
import org.javaup.ai.enums.BaseCode;
import org.javaup.ai.vo.TicketCategoryDetailVo;
import org.javaup.ai.vo.result.TicketCategoryListResultVo;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;

import static org.javaup.ai.constants.DaMaiConstant.TICKET_LIST_URL;


@Component
public class TicketCategoryCall {

    // 按节目 id 查询票档明细。
    // 这里拿到的不只是票档基础信息，还包括库存等更适合下单前校验的数据。
    public List<TicketCategoryDetailVo> selectListByProgram(TicketCategoryListByProgramDto ticketCategoryListByProgramDto) {
        // 调用后端票档接口，把节目 id 作为查询条件发给业务系统。
        String result = HttpRequest.post(TICKET_LIST_URL)
                .header("no_verify", "true")
                .body(JSON.toJSONString(ticketCategoryListByProgramDto))
                .timeout(20000)
                .execute().body();
        // 把返回结果转换成统一的票档结果对象。
        TicketCategoryListResultVo ticketCategoryListResultVo = JSON.parseObject(result, TicketCategoryListResultVo.class);
        // 如果后端返回失败码，说明票档接口调用失败，直接中断流程。
        if (!Objects.equals(ticketCategoryListResultVo.getCode(), BaseCode.SUCCESS.getCode())) {
            throw new RuntimeException("调用演出票务系统查询票档信息失败");
        }
        // 如果没有任何票档数据，说明当前节目不可售或票档信息异常，也不允许继续往后走。
        if (CollectionUtil.isEmpty(ticketCategoryListResultVo.getData())) {
            throw new RuntimeException("票档信息不存在");
        }
        // 返回真实票档明细，供上层做库存展示、票档映射和下单校验。
        return ticketCategoryListResultVo.getData();
    }
}
