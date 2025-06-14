<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "bbce3572338711aeab758506379ab716",
  "translation_date": "2025-06-11T04:49:34+00:00",
  "source_file": "11-mcp/README.md",
  "language_code": "ja"
}
-->
# Lesson 11: Model Context Protocol (MCP) Integration

## Introduction to Model Context Protocol (MCP)

Model Context Protocol (MCP) は、AIモデルとクライアントアプリケーション間のやり取りを標準化するために設計された最先端のフレームワークです。MCPはAIモデルとそれを利用するアプリケーションの橋渡し役を果たし、基盤となるモデルの実装に関わらず一貫したインターフェースを提供します。

MCPの主な特徴：

- **標準化された通信**：アプリケーションがAIモデルと共通の言語でやり取りできるようにします
- **強化されたコンテキスト管理**：AIモデルへ効率的にコンテキスト情報を渡せます
- **クロスプラットフォーム対応**：C#、Java、JavaScript、Python、TypeScriptなど様々な言語で動作します
- **シームレスな統合**：開発者が異なるAIモデルを簡単にアプリに組み込めるようにします

MCPは特にAIエージェント開発において、統一されたプロトコルを通じて多様なシステムやデータソースとやり取りできるため、エージェントの柔軟性と性能を高めるのに役立ちます。

## Learning Objectives
- MCPとは何か、そのAIエージェント開発における役割を理解する
- GitHub連携のためにMCPサーバーをセットアップ・構成する
- MCPツールを使ってマルチエージェントシステムを構築する
- Azure Cognitive Searchを用いたRAG（Retrieval Augmented Generation）を実装する

## Prerequisites
- Python 3.8以上
- Node.js 14以上
- Azureサブスクリプション
- GitHubアカウント
- Semantic Kernelの基本知識

## Setup Instructions

1. **環境セットアップ**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Azureサービスの構成**
   - Azure Cognitive Searchリソースの作成
   - Azure OpenAIサービスの設定
   - `.env`ファイルに環境変数を設定

3. **MCPサーバーのセットアップ**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Project Structure

```
11-mcp/
├── code_samples/
│   └── github-mcp/
│       ├── app.py              # Main application
│       ├── event-descriptions.md  # Event data
│       └── MCP_SETUP.md        # Setup guide
├── README.md
└── requirements.txt
```

## Core Components

### 1. マルチエージェントシステム
- GitHubエージェント：リポジトリ分析
- ハッカソンエージェント：プロジェクト推薦
- イベントエージェント：技術イベントの提案

### 2. Azure連携
- イベントインデックス用のCognitive Search
- エージェント知能用のAzure OpenAI
- RAGパターンの実装

### 3. MCPツール
- GitHubリポジトリ分析
- コード検査
- メタデータ抽出

## Code Walkthrough

このサンプルでは以下を示します：
1. MCPサーバーの統合
2. マルチエージェントの調整
3. Azure Cognitive Searchの連携
4. RAGパターンの実装

主な機能：
- リアルタイムGitHubリポジトリ分析
- インテリジェントなプロジェクト推薦
- Azure Searchを用いたイベントマッチング
- Chainlitによるストリーミングレスポンス

## Running the Sample

詳細なセットアップ手順や追加情報は、[Github MCP Server Example README](./code_samples/github-mcp/README.md)を参照してください。

1. MCPサーバーを起動：
   ```bash
   npx @modelcontextprotocol/server-github
   ```

2. アプリケーションを起動：
   ```bash
   chainlit run app.py -w
   ```

3. 統合をテスト：
   ```
   Example query: "Analyze repositories for username: <github_username>"
   ```

## Troubleshooting

よくある問題と解決策：
1. MCP接続問題
   - サーバーが起動しているか確認
   - ポートの使用状況をチェック
   - GitHubトークンを確認

2. Azure Searchの問題
   - 接続文字列の検証
   - インデックスの存在確認
   - ドキュメントのアップロード確認

## Next Steps
- さらなるMCPツールの探索
- カスタムエージェントの実装
- RAG機能の強化
- イベントソースの追加

## Resources
- [MCP for Beginners](https://aka.ms/mcp-for-beginners)  
- [MCP Documentation](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic-kernel/semantic_kernel/connectors/mcp)
- [Azure Cognitive Search Docs](https://learn.microsoft.com/azure/search/)
- [Semantic Kernel Guides](https://learn.microsoft.com/semantic-kernel/)

**免責事項**：  
本書類はAI翻訳サービス「Co-op Translator」（https://github.com/Azure/co-op-translator）を使用して翻訳されています。正確性を期しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知おきください。原文の母国語版が正式な情報源とみなされます。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じた誤解や誤訳について、当方は一切の責任を負いかねます。