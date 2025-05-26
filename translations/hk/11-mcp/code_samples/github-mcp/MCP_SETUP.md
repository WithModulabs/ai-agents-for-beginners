<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:14:51+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "hk"
}
-->
# MCP Server 集成指南

## 前置條件
- 已安裝 Node.js（版本 14 或以上）
- npm 套件管理器
- 配備所需依賴的 Python 環境

## 設置步驟

1. **安裝 MCP Server 套件**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **啟動 MCP Server**
   ```bash
   npx @modelcontextprotocol/server-github
   ```
   伺服器應該會啟動並顯示連接 URL。

3. **驗證連接**
   - 在你的 Chainlit 介面中尋找插頭圖示 (🔌)
   - 插頭圖示旁應該會出現數字 (1)，表示連接成功
   - 主控台應顯示：「GitHub plugin setup completed successfully」（以及其他狀態訊息）

## 疑難排解

### 常見問題

1. **埠口衝突**
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```
   解決方法：使用以下指令更改埠口：
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **認證問題**
   - 確認 GitHub 憑證已正確設定
   - 檢查 .env 檔案是否包含所需的 Token
   - 驗證 GitHub API 訪問權限

3. **連接失敗**
   - 確認伺服器在預期的埠口運行
   - 檢查防火牆設定
   - 確認 Python 環境已安裝所需套件

## 連接驗證

當你的 MCP Server 正常連接時：
1. 主控台會顯示「GitHub plugin setup completed successfully」
2. 連接日誌會顯示「✓ MCP Connection Status: Active」
3. GitHub 指令可在聊天介面正常使用

## 環境變數

在你的 .env 檔案中需包含：
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## 測試連接

在聊天中發送此測試訊息：
```
Show me the repositories for username: [GitHub Username]
```
成功回應會顯示倉庫資訊。

**免責聲明**：  
本文件係使用AI翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我哋努力確保準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原文文件以其原生語言版本為權威來源。對於重要資訊，建議使用專業人工翻譯。對於因使用本翻譯而引起嘅任何誤解或誤譯，我哋概不負責。