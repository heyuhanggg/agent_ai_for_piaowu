# 配置说明文档

## 环境准备

### 1. 必需软件
- JDK 17+
- Maven 3.6+
- MySQL 8.0+
- Elasticsearch 8.x
- Ollama (可选，用于本地模型)

### 2. 数据库初始化
执行 `sql/damai_ai.sql` 脚本创建数据库和表结构：
```bash
mysql -u root -p < sql/damai_ai.sql
```

### 3. 配置文件设置

#### 3.1 复制示例配置文件
```bash
# 核心服务配置
cp damai-core-service/src/main/resources/application.yaml.example \
   damai-core-service/src/main/resources/application.yaml

# MCP日志服务配置（如果有）
cp damai-mcp-server/damai-mcp-log-service/src/main/resources/application.yaml.example \
   damai-mcp-server/damai-mcp-log-service/src/main/resources/application.yaml

# MCP指标服务配置（如果有）
cp damai-mcp-server/damai-mcp-metrics-service/src/main/resources/application.yaml.example \
   damai-mcp-server/damai-mcp-metrics-service/src/main/resources/application.yaml
```

#### 3.2 配置环境变量

**方式一：系统环境变量（推荐）**

Windows:
```powershell
# 设置阿里云API密钥
[Environment]::SetEnvironmentVariable("alibaba-key", "YOUR_ALIBABA_API_KEY", "User")

# 设置DeepSeek API密钥
[Environment]::SetEnvironmentVariable("deepseek-key", "YOUR_DEEPSEEK_API_KEY", "User")
```

Linux/Mac:
```bash
export alibaba-key="YOUR_ALIBABA_API_KEY"
export deepseek-key="YOUR_DEEPSEEK_API_KEY"

# 永久保存，添加到 ~/.bashrc 或 ~/.zshrc
echo 'export alibaba-key="YOUR_ALIBABA_API_KEY"' >> ~/.bashrc
echo 'export deepseek-key="YOUR_DEEPSEEK_API_KEY"' >> ~/.bashrc
```

**方式二：创建 .env 文件（已在.gitignore中排除）**

在项目根目录创建 `.env` 文件：
```properties
# AI模型API密钥
alibaba-key=YOUR_ALIBABA_API_KEY
deepseek-key=YOUR_DEEPSEEK_API_KEY

# 数据库配置
MYSQL_PASSWORD=root
ELASTICSEARCH_PASSWORD=elastic
```

#### 3.3 修改 application.yaml

修改以下配置项：

1. **MySQL数据库**
```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/damai_ai?...
    username: root
    password: YOUR_MYSQL_PASSWORD  # 修改为实际密码
```

2. **Elasticsearch**
```yaml
easy-es:
  address: 127.0.0.1:9200
  username: elastic
  password: YOUR_ELASTICSEARCH_PASSWORD  # 修改为实际密码
```

3. **AI模型密钥**
```yaml
spring:
  ai:
    openai:
      api-key: ${alibaba-key}  # 从环境变量读取
    deepseek:
      api-key: ${deepseek-key}  # 从环境变量读取
```

### 4. 获取API密钥

#### 阿里云百炼 (Qwen)
1. 访问 [阿里云百炼控制台](https://dashscope.aliyuncs.com/)
2. 注册/登录账号
3. 进入 API-KEY 管理页面创建密钥
4. 复制 API Key 设置到环境变量 `alibaba-key`

#### DeepSeek
1. 访问 [DeepSeek 官网](https://platform.deepseek.com/)
2. 注册/登录账号
3. 进入 API Keys 页面创建密钥
4. 复制 API Key 设置到环境变量 `deepseek-key`

### 5. RAG版本配置

项目支持两种RAG实现方式，通过环境变量控制：

```yaml
# V1: 使用 QuestionAnswerAdvisor（默认）
rag.version=1

# V2: 使用混合检索 + Rerank
rag.version=2
```

### 6. MCP配置（可选）

如果需要使用运维助手功能，配置MCP服务：

```yaml
spring:
  ai:
    mcp:
      client:
        sse:
          connections:
            damai-log:
              url: http://localhost:8085
            damai-metrics:
              url: http://localhost:8086
```

## 启动项目

### 1. 编译打包
```bash
mvn clean package -DskipTests
```

### 2. 启动核心服务
```bash
java -jar damai-core-service/target/damai-core-service-*.jar
```

### 3. 启动前端（如果有）
```bash
cd vue
npm install
npm run dev
```

## 常见问题

### 1. API密钥未设置
**错误**: `Could not resolve placeholder 'alibaba-key'`

**解决**: 确保已设置环境变量或在application.yaml中直接配置

### 2. 数据库连接失败
**解决**: 
- 检查MySQL是否启动
- 验证数据库名称、用户名、密码是否正确
- 确认数据库已执行初始化脚本

### 3. Elasticsearch连接失败
**解决**:
- 检查ES是否启动
- 验证地址和端口是否正确
- 如果不需要ES功能，可以禁用：`easy-es.enable=false`

## 项目架构

- **LLM模型**: DeepSeek (贴心助手、运维助手、标题生成)
- **LLM模型**: Qwen Max (规则助手RAG)
- **Embedding模型**: OpenAI格式的text-embedding-v3
- **向量数据库**: SimpleVectorStore (内存)
- **关系数据库**: MySQL (会话历史、观测数据)
- **全文检索**: Elasticsearch (可选)

## 安全注意事项

⚠️ **重要**: 
- **不要将 application.yaml 提交到Git**
- **不要在代码中硬编码API密钥**
- **使用环境变量或密钥管理服务**
- **定期轮换API密钥**
