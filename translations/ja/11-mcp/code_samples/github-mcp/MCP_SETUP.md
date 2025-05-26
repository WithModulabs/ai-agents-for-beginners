<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:15:19+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "ja"
}
-->
# MCPサーバー統合ガイド

## 前提条件
- Node.jsがインストールされていること（バージョン14以上）
- npmパッケージマネージャー
- 必要な依存関係が整ったPython環境

## セットアップ手順

1. **MCPサーバーパッケージをインストール**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **MCPサーバーを起動**
   ```bash
   npx @modelcontextprotocol/server-github
   ```
   サーバーが起動し、接続用のURLが表示されます。

3. **接続の確認**
   - Chainlitインターフェースにプラグアイコン（🔌）が表示されているか確認
   - プラグアイコンの横に数字（1）が表示されていれば接続成功です
   - コンソールには「GitHub plugin setup completed successfully」と表示され（追加のステータス行も含む）、正常にセットアップされています

## トラブルシューティング

### よくある問題

1. **ポートの競合**
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```
   対処法：以下のコマンドでポートを変更してください
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **認証の問題**
   - GitHubの認証情報が正しく設定されているか確認
   - .envファイルに必要なトークンが含まれているかチェック
   - GitHub APIへのアクセス権を確認

3. **接続失敗**
   - サーバーが指定したポートで動作しているか確認
   - ファイアウォールの設定を確認
   - Python環境に必要なパッケージがインストールされているか確認

## 接続確認

MCPサーバーが正しく接続されている状態は以下の通りです：
1. コンソールに「GitHub plugin setup completed successfully」と表示される
2. 接続ログに「✓ MCP Connection Status: Active」と表示される
3. チャットインターフェースでGitHubコマンドが正常に動作する

## 環境変数

.envファイルに必須の設定：
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## 接続テスト

チャットで以下のテストメッセージを送信してください：
```
Show me the repositories for username: [GitHub Username]
```
正常に応答が返るとリポジトリ情報が表示されます。

**免責事項**:  
本書類はAI翻訳サービス「[Co-op Translator](https://github.com/Azure/co-op-translator)」を使用して翻訳されています。正確性には努めておりますが、自動翻訳には誤りや不正確な箇所が含まれる可能性があることをご了承ください。原文の言語によるオリジナル文書が正式な情報源とみなされます。重要な情報については、専門の人間翻訳をご利用いただくことを推奨します。本翻訳の使用により生じた誤解や誤訳について、一切の責任を負いかねます。