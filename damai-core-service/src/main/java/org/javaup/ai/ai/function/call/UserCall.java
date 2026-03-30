package org.javaup.ai.ai.function.call;

import cn.hutool.http.HttpRequest;
import com.alibaba.fastjson.JSON;
import org.javaup.ai.enums.BaseCode;
import org.javaup.ai.vo.TicketUserVo;
import org.javaup.ai.vo.UserDetailVo;
import org.javaup.ai.vo.result.TicketUserResultVo;
import org.javaup.ai.vo.result.UserDetailResultVo;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.javaup.ai.constants.DaMaiConstant.TICKET_USER_LIST_URL;
import static org.javaup.ai.constants.DaMaiConstant.USER_DETAIL_URL;


@Component
public class UserCall {
    
    // 根据手机号查询用户主档信息。
    // 这是把用户在对话里提供的手机号映射成后端真实 userId 的关键一步。
    public UserDetailVo userDetail(String mobile){
        // 组装后端接口参数，当前接口只需要手机号。
        Map<String,String> params = new HashMap<>(2);
        params.put("mobile", mobile);
        UserDetailResultVo userDetailResultVo = new UserDetailResultVo();
        // 调用用户查询接口，拿真实用户信息。
        String result = HttpRequest.post(USER_DETAIL_URL)
                .header("no_verify", "true")
                .body(JSON.toJSONString(params))
                .timeout(20000)
                .execute().body();
        // 反序列化接口返回结果。
        userDetailResultVo = JSON.parseObject(result, UserDetailResultVo.class);
        // 返回失败码时直接抛错，避免继续走下单流程。
        if (!Objects.equals(userDetailResultVo.getCode(), BaseCode.SUCCESS.getCode())) {
            throw new RuntimeException("调用演出票务系统查询用户信息失败");
        }
        // 返回 data 中的用户详情对象，上层会继续提取 userId。
        return userDetailResultVo.getData();
    }
    
    // 根据用户 id 查询该用户维护的全部购票人。
    // 购票人列表是下单时做身份匹配和生成 ticketUserIdList 的基础数据。
    public List<TicketUserVo> ticketUserList(Long userId){
        // 组装查询购票人接口的参数。
        Map<String,Object> params = new HashMap<>(2);
        params.put("userId", userId);
        TicketUserResultVo ticketUserResultVo = new TicketUserResultVo();
        // 调用后端购票人列表接口。
        String result = HttpRequest.post(TICKET_USER_LIST_URL)
                .header("no_verify", "true")
                .body(JSON.toJSONString(params))
                .timeout(20000)
                .execute().body();
        // 把返回结果解析成统一对象。
        ticketUserResultVo = JSON.parseObject(result, TicketUserResultVo.class);
        // 如果接口执行失败，直接抛错。
        if (!Objects.equals(ticketUserResultVo.getCode(), BaseCode.SUCCESS.getCode())) {
            throw new RuntimeException("调用演出票务系统查询购票人信息失败");
        }
        // 如果 data 为空，说明当前用户没有可用购票人信息。
        if (Objects.isNull(ticketUserResultVo.getData())) {
            throw new RuntimeException("购票人信息不存在");
        }
        // 返回购票人列表给上层做证件号匹配。
        return ticketUserResultVo.getData();
    }
}
