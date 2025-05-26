<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:11:50+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "ja"
}
-->
# Github MCP サーバーの例

## 説明

これは Microsoft Reactor が主催する AI Agents ハッカソンのために作成されたデモです。

このツールは、ユーザーの Github リポジトリに基づいてハッカソンプロジェクトを推薦します。
方法は以下の通りです：

1. **Github Agent** - Github MCP サーバーを使ってリポジトリとその情報を取得します。
2. **Hackathon Agent** - Github Agent から得たデータをもとに、ユーザーのプロジェクトや使用言語、AI Agents ハッカソンのプロジェクトトラックに基づいた創造的なハッカソンプロジェクトのアイデアを考え出します。
3. **Events Agent** - Hackathon Agent の提案をもとに、AI Agent ハッカソンシリーズの関連イベントを推薦します。

## コードの実行

### 環境変数

このデモでは Azure Open AI Service、Semantic Kernel、Github MCP サーバー、Azure AI Search を使用します。

これらのツールを使うために、適切な環境変数が設定されていることを確認してください：

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
``` 

## Chainlit サーバーの起動

MCP サーバーに接続するために、このデモでは Chainlit をチャットインターフェースとして使用します。

サーバーを起動するには、ターミナルで以下のコマンドを使います：

```bash
chainlit run app.py -w
```

これで `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md` 上に Chainlit サーバーが起動します。

## MCP サーバーへの接続

Github MCP サーバーに接続するには、「Type your message here..」チャットボックスの下にある「プラグ」アイコンを選択してください：

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

そこから「Connect an MCP」をクリックして、Github MCP サーバーへの接続コマンドを追加できます：

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

"[YOUR PERSONAL ACCESS TOKEN]" を実際の Personal Access Token に置き換えてください。

接続が成功すると、プラグアイコンの横に (1) が表示されて接続が確認できます。表示されない場合は、`chainlit run app.py -w` で chainlit サーバーを再起動してみてください。

## デモの使い方

ハッカソンプロジェクトの推薦エージェントワークフローを開始するには、次のようなメッセージを入力します：

"Recommend hackathon projects for the Github user koreyspace"

Router Agent がリクエストを分析し、どのエージェント（GitHub、Hackathon、Events）の組み合わせが最適か判断します。エージェントは連携して、Github リポジトリの分析、プロジェクトのアイデア出し、関連する技術イベントに基づいた総合的な推薦を提供します。

**免責事項**：  
本書類はAI翻訳サービス「[Co-op Translator](https://github.com/Azure/co-op-translator)」を使用して翻訳されています。正確性には努めておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご理解ください。原文の言語で記載された文書が正式な情報源となります。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や誤訳についても、当方は責任を負いかねます。