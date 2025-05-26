<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:14:35+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "zh"
}
-->
# MCP 服务器集成指南

## 前提条件
- 已安装 Node.js（版本 14 或更高）
- npm 包管理器
- 配备所需依赖的 Python 环境

## 设置步骤

1. **安装 MCP Server 包**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **启动 MCP Server**
   ```bash
   npx @modelcontextprotocol/server-github
   ```  
   服务器启动后会显示一个连接 URL。

3. **验证连接**
   - 在 Chainlit 界面中查找插头图标 (🔌)  
   - 插头图标旁应显示数字 (1)，表示连接成功  
   - 控制台应显示：“GitHub plugin setup completed successfully”（以及其他状态信息）

## 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```  
   解决方案：使用以下命令更改端口：  
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **身份验证问题**
   - 确保 GitHub 凭据配置正确  
   - 检查 .env 文件中包含所需的令牌  
   - 验证 GitHub API 访问权限

3. **连接失败**
   - 确认服务器运行在预期端口  
   - 检查防火墙设置  
   - 确认 Python 环境已安装所需包

## 连接验证

当满足以下条件时，您的 MCP 服务器连接正常：  
1. 控制台显示 “GitHub plugin setup completed successfully”  
2. 连接日志显示 “✓ MCP Connection Status: Active”  
3. 聊天界面中 GitHub 命令可正常使用

## 环境变量

请在您的 .env 文件中添加：  
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## 测试连接

在聊天中发送此测试消息：  
```
Show me the repositories for username: [GitHub Username]
```  
成功响应将显示仓库信息。

**免责声明**：  
本文件使用 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。虽然我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始语言版本的文件应被视为权威来源。对于重要信息，建议采用专业人工翻译。我们不对因使用本翻译而产生的任何误解或误释承担责任。