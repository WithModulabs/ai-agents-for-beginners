<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:11:03+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "hk"
}
-->
# Github MCP Server 範例

## 描述

呢個係為咗 Microsoft Reactor 主辦嘅 AI Agents Hackathon 而整嘅示範。

呢個工具用嚟根據用戶嘅 Github 倉庫推介 hackathon 項目。
做法係：

1. **Github Agent** - 用 Github MCP Server 去攞倉庫同埋有關呢啲倉庫嘅資料。
2. **Hackathon Agent** - 由 Github Agent 攞返嘅資料，根據用戶嘅項目、用嘅語言同埋 AI Agents hackathon 嘅項目類別，諗出有創意嘅 hackathon 項目點子。
3. **Events Agent** - 根據 hackathon agent 嘅建議，events agent 會推介 AI Agent Hackathon 系列嘅相關活動。

## 運行程式碼

### 環境變數

呢個示範用咗 Azure Open AI Service、Semantic Kernel、Github MCP Server 同 Azure AI Search。

記住要設定好啱嘅環境變數先可以用呢啲工具：

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
```

## 運行 Chainlit Server

為咗連接 MCP server，呢個示範用 Chainlit 做聊天介面。

喺終端機用以下指令運行 server：

```bash
chainlit run app.py -w
```

咁就會喺 `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md` 開啟你嘅 Chainlit server。

## 連接 MCP Server

要連接 Github MCP Server，喺「Type your message here..」嘅聊天框下面揀「插頭」圖示：

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

之後你可以撳「Connect an MCP」加指令去連接 Github MCP Server：

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

將 "[YOUR PERSONAL ACCESS TOKEN]" 換成你自己嘅 Personal Access Token。

連接成功後，插頭圖示旁邊會見到 (1) 嘅標示，代表已連接。如果冇，試下用 `chainlit run app.py -w` 重啟 chainlit server。

## 使用示範

要開始推薦 hackathon 項目嘅代理工作流程，你可以打句似咁嘅訊息：

"Recommend hackathon projects for the Github user koreyspace"

Router Agent 會分析你嘅請求，決定用邊啲代理（GitHub、Hackathon 同 Events）嘅組合最啱處理你嘅查詢。呢啲代理會一齊合作，根據 GitHub 倉庫分析、項目構思同相關技術活動，提供全面嘅推介。

**免責聲明**：  
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原文文件的母語版本應被視為權威來源。對於重要資訊，建議使用專業人工翻譯。我們不會對因使用此翻譯而引起的任何誤解或誤釋負責。