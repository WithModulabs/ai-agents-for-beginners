<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:14:42+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "tw"
}
-->
# MCP Server 整合指南

## 前置條件
- 已安裝 Node.js（版本 14 或以上）
- npm 套件管理工具
- 含必要相依套件的 Python 環境

## 設定步驟

1. **安裝 MCP Server 套件**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **啟動 MCP Server**
   ```bash
   npx @modelcontextprotocol/server-github
   ```
   伺服器啟動後會顯示連線 URL。

3. **驗證連線**
   - 在你的 Chainlit 介面中尋找插頭圖示 (🔌)
   - 插頭圖示旁應該會顯示數字 (1)，代表連線成功
   - 主控台會顯示：「GitHub plugin setup completed successfully」（以及其他狀態訊息）

## 疑難排解

### 常見問題

1. **埠口衝突**
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```
   解決方案：使用以下指令更換埠口：
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **認證問題**
   - 確認 GitHub 認證資料已正確設定
   - 檢查 .env 檔案是否包含必要的 token
   - 驗證 GitHub API 存取權限

3. **連線失敗**
   - 確認伺服器正在預期的埠口上執行
   - 檢查防火牆設定
   - 確認 Python 環境已安裝必要套件

## 連線驗證

當你的 MCP 伺服器成功連線時：
1. 主控台會顯示「GitHub plugin setup completed successfully」
2. 連線日誌會顯示「✓ MCP Connection Status: Active」
3. GitHub 指令能在聊天介面正常使用

## 環境變數

請在你的 .env 檔案中設定：
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## 測試連線

在聊天中傳送此測試訊息：
```
Show me the repositories for username: [GitHub Username]
```
成功回應會顯示倉庫資訊。

**免責聲明**：  
本文件係使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意，自動翻譯可能包含錯誤或不精確之處。原始文件的母語版本應視為權威來源。對於重要資訊，建議採用專業人工翻譯。我們不對因使用本翻譯所產生之任何誤解或誤釋負責。