package org.javaup.ai;

import org.dromara.easyes.spring.annotation.EsMapperScan;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@EsMapperScan("org.javaup.ai.es.mapper")
@MapperScan("org.javaup.ai.mapper")
@SpringBootApplication
public class DaMaiAiCoreApplication {

    public static void main(String[] args) {
        SpringApplication.run(DaMaiAiCoreApplication.class, args);
    }

}
