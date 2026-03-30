# GitHub部署指南

## 准备工作

### 1. 在GitHub创建新仓库
1. 访问 https://github.com/new
2. 填写仓库名称，例如：`damai-ai`
3. 选择 Public（公开）或 Private（私有）
4. **不要**勾选 "Initialize this repository with a README"
5. 点击 "Create repository"

### 2. 本地Git初始化和推送

打开PowerShell，切换到项目目录：
```powershell
cd e:\java\code\damai-ai-master\damai-ai-master
```

#### 步骤1: 初始化Git仓库（如果还未初始化）
```powershell
git init
```

#### 步骤2: 添加所有文件
```powershell
git add .
```

#### 步骤3: 检查将要提交的文件
```powershell
# 确保 application.yaml 不在列表中
git status
```

⚠️ **重要检查**: 确保以下文件**不在**待提交列表中：
- `application.yaml`
- `.env`
- 任何包含密码/API密钥的文件

如果看到敏感文件，执行：
```powershell
git rm --cached damai-core-service/src/main/resources/application.yaml
```

#### 步骤4: 提交代码
```powershell
git commit -m "Initial commit: 演出票务AI智能服务系统"
```

#### 步骤5: 关联远程仓库
```powershell
# 替换 YOUR_USERNAME 为你的GitHub用户名
# 替换 damai-ai 为你的仓库名
git remote add origin https://github.com/YOUR_USERNAME/damai-ai.git
```

#### 步骤6: 推送到GitHub
```powershell
# 推送到main分支
git branch -M main
git push -u origin main
```

如果遇到认证问题，使用Personal Access Token：
```powershell
# GitHub不再支持密码认证，需要使用Token
# 访问 https://github.com/settings/tokens 创建Token
# 权限选择: repo (完整控制私有仓库)
```

### 3. 后续更新代码

```powershell
# 添加修改的文件
git add .

# 提交更改
git commit -m "描述你的修改内容"

# 推送到远程
git push
```

## 常见问题

### 1. 如果已经提交了敏感信息怎么办？

**立即从历史记录中删除**：
```powershell
# 从历史记录中删除文件
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch damai-core-service/src/main/resources/application.yaml" --prune-empty --tag-name-filter cat -- --all

# 强制推送（危险操作，确保你知道在做什么）
git push origin --force --all
```

**更安全的方式 - 使用BFG Repo-Cleaner**：
```powershell
# 下载 BFG: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --delete-files application.yaml
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push origin --force --all
```

**重要**: 如果API密钥已泄露，立即前往对应平台删除并重新生成！

### 2. 克隆仓库后如何配置？

其他开发者克隆仓库后：
```powershell
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/damai-ai.git
cd damai-ai

# 复制示例配置
cp damai-core-service/src/main/resources/application.yaml.example `
   damai-core-service/src/main/resources/application.yaml

# 编辑配置文件，填入实际的密码和API密钥
# 详见 CONFIG.md
```

### 3. 分支管理建议

```powershell
# 创建开发分支
git checkout -b develop

# 创建功能分支
git checkout -b feature/new-feature

# 合并到主分支
git checkout main
git merge develop

# 删除已合并的分支
git branch -d feature/new-feature
```

## README更新建议

建议在现有 README.md 中添加以下内容：

```markdown
## 快速开始

### 1. 克隆项目
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/damai-ai.git
cd damai-ai
\`\`\`

### 2. 配置环境
详见 [CONFIG.md](CONFIG.md) 获取完整配置说明

### 3. 启动项目
\`\`\`bash
mvn clean package
java -jar damai-core-service/target/damai-core-service-*.jar
\`\`\`

## 环境要求
- JDK 17+
- Maven 3.6+
- MySQL 8.0+
- Elasticsearch 8.x (可选)

## 配置说明
⚠️ 本项目需要配置以下API密钥：
- 阿里云百炼 API Key (Qwen模型)
- DeepSeek API Key
- MySQL数据库密码
- Elasticsearch密码（可选）

详细配置步骤请参考 [CONFIG.md](CONFIG.md)
\`\`\`

## License
本项目采用 [LICENSE](LICENSE) 许可证
```

## 安全最佳实践

✅ **应该做的**:
- 使用环境变量存储敏感信息
- 定期更新 `.gitignore`
- 提交前检查 `git status`
- 使用 `application.yaml.example` 作为模板
- 在 README 中说明需要配置的环境变量

❌ **不应该做的**:
- 将真实的API密钥提交到Git
- 将数据库密码硬编码在代码中
- 在公开仓库中暴露内网地址
- 提交生产环境配置文件
