<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-06-11T04:50:54+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "ja"
}
-->
# Github MCPサーバーの例

## 説明

これはMicrosoft Reactor主催のAI Agentsハッカソン向けに作成されたデモです。

このツールは、ユーザーのGithubリポジトリに基づいてハッカソンプロジェクトを推薦します。具体的には以下のように行います：

1. **Github Agent** - Github MCPサーバーを使ってリポジトリやその情報を取得します。
2. **Hackathon Agent** - Github Agentから得たデータをもとに、ユーザーのプロジェクトや使用言語、AI Agentsハッカソンのプロジェクトトラックに基づいた創造的なハッカソンプロジェクトのアイデアを考案します。
3. **Events Agent** - Hackathon Agentの提案をもとに、AI Agentハッカソンシリーズの関連イベントを推薦します。

## コードの実行

### 環境変数

このデモではAzure Open AI Service、Semantic Kernel、Github MCPサーバー、Azure AI Searchを使用しています。

これらのツールを利用するために、適切な環境変数が設定されていることを確認してください：

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
``` 

## Chainlitサーバーの起動

MCPサーバーに接続するため、このデモではChainlitをチャットインターフェースとして使用しています。

サーバーを起動するには、ターミナルで以下のコマンドを実行してください：

```bash
chainlit run app.py -w
```

これにより、`localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md` でChainlitサーバーが起動します。

## MCPサーバーへの接続

Github MCPサーバーに接続するには、「Type your message here..」チャットボックスの下にある「プラグ」アイコンを選択してください：

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

次に、「Connect an MCP」をクリックしてGithub MCPサーバーへの接続コマンドを追加します：

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

"[YOUR PERSONAL ACCESS TOKEN]" は実際のパーソナルアクセストークンに置き換えてください。

接続が完了すると、プラグアイコンの横に(1)が表示されて接続が確認できます。表示されない場合は、`chainlit run app.py -w`でChainlitサーバーを再起動してみてください。

## デモの使い方

ハッカソンプロジェクトの推薦ワークフローを開始するには、例えば以下のように入力します：

"Recommend hackathon projects for the Github user koreyspace"

Router Agentがリクエストを解析し、どのエージェント（Github、Hackathon、Events）の組み合わせが最適か判断します。エージェントは連携して、Githubリポジトリの分析、プロジェクトのアイデア出し、関連技術イベントの推薦を総合的に提供します。

**免責事項**：  
本書類はAI翻訳サービス「Co-op Translator」（https://github.com/Azure/co-op-translator）を使用して翻訳されました。正確性を期しておりますが、自動翻訳には誤りや不正確な箇所が含まれる可能性があることをご承知おきください。原文はその言語における正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や誤訳についても、当方は責任を負いかねます。