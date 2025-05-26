<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:10:23+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "zh"
}
-->
# Github MCP 服务器示例

## 描述

这是为微软 Reactor 举办的 AI Agents 黑客松创建的一个演示。

该工具用于根据用户的 Github 仓库推荐黑客松项目。
具体流程如下：

1. **Github Agent** - 使用 Github MCP 服务器获取仓库及其相关信息。
2. **Hackathon Agent** - 根据 Github Agent 提供的数据，结合用户使用的编程语言和 AI Agents 黑客松的项目方向，提出有创意的黑客松项目想法。
3. **Events Agent** - 根据 Hackathon Agent 的建议，推荐 AI Agent 黑客松系列中的相关活动。

## 运行代码

### 环境变量

此演示使用了 Azure Open AI 服务、Semantic Kernel、Github MCP 服务器和 Azure AI 搜索。

请确保已正确设置环境变量以使用这些工具：

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
```

## 运行 Chainlit 服务器

为了连接 MCP 服务器，此演示使用 Chainlit 作为聊天界面。

在终端中运行以下命令启动服务器：

```bash
chainlit run app.py -w
```

这将启动你的 Chainlit 服务器，地址为 `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md` 内容。

## 连接 MCP 服务器

要连接到 Github MCP 服务器，请点击“Type your message here..”聊天框下方的“插头”图标：

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

然后点击“Connect an MCP”以添加连接 Github MCP 服务器的命令：

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

将 “[YOUR PERSONAL ACCESS TOKEN]” 替换为你的个人访问令牌。

连接成功后，插头图标旁应显示 (1) 以确认连接。如果未显示，请尝试使用 `chainlit run app.py -w` 重启 Chainlit 服务器。

## 使用演示

要启动推荐黑客松项目的代理工作流，可以输入类似以下内容：

“Recommend hackathon projects for the Github user koreyspace”

Router Agent 会分析你的请求，判断使用哪种代理组合（Github、Hackathon 和 Events）最适合处理你的查询。各代理协同工作，基于 Github 仓库分析、项目创意和相关技术活动，为你提供全面的推荐。

**免责声明**：  
本文件由AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)翻译。尽管我们力求准确，但请注意，自动翻译可能存在错误或不准确之处。原始语言版本的文件应被视为权威来源。对于重要信息，建议采用专业人工翻译。对于因使用本翻译而产生的任何误解或错误解释，我们不承担任何责任。