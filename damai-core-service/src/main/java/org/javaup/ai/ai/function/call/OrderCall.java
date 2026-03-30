package org.javaup.ai.ai.function.call;

import cn.hutool.http.HttpRequest;
import com.alibaba.fastjson.JSON;
import org.javaup.ai.dto.ProgramOrderCreateDto;
import org.javaup.ai.enums.BaseCode;
import org.javaup.ai.vo.result.CreateOrderResult;
import org.springframework.stereotype.Component;

import java.util.Objects;

import static org.javaup.ai.constants.DaMaiConstant.CREATE_ORDER_URL;


@Component
public class OrderCall {
    
    // 调用后端真实下单接口创建订单。
    // 这里接收的已经不是模型原始参数，而是服务端校验并映射完成后的标准订单 DTO。
    public String createOrder(ProgramOrderCreateDto programOrderCreateDto){
        CreateOrderResult createOrderResult = new CreateOrderResult();
        // 把标准订单参数发送到后端创建订单接口。
        String result = HttpRequest.post(CREATE_ORDER_URL)
                .header("no_verify", "true")
                .body(JSON.toJSONString(programOrderCreateDto))
                .timeout(20000)
                .execute().body();
        // 解析后端下单结果，判断是否创建成功。
        createOrderResult = JSON.parseObject(result, CreateOrderResult.class);
        // 只要返回码不是成功，就视为下单失败，抛异常给上层处理。
        if (!Objects.equals(createOrderResult.getCode(), BaseCode.SUCCESS.getCode())) {
            throw new RuntimeException("调用演出票务系统创建订单失败");
        }
        // 返回后端生成的订单号，供 AI 工具层再封装给前端。
        return createOrderResult.getData();
    }
}
