<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "f0ce2d470f3efad6f8c7df376f416a4b",
  "translation_date": "2025-06-11T04:53:00+00:00",
  "source_file": "00-course-setup/AzureSearch.md",
  "language_code": "ja"
}
-->
# Azure AI Search セットアップガイド

このガイドでは、Azure ポータルを使って Azure AI Search を設定する方法をご案内します。以下の手順に従って、Azure AI Search サービスを作成および構成してください。

## 前提条件

開始する前に、以下を準備してください：

- Azure サブスクリプション。まだお持ちでない場合は、[Azure Free Account](https://azure.microsoft.com/free/?wt.mc_id=studentamb_258691) で無料アカウントを作成できます。

## ステップ 1: Azure AI Search サービスの作成

1. [Azure ポータル](https://portal.azure.com/?wt.mc_id=studentamb_258691)にサインインします。
2. 左側のナビゲーションペインで、**リソースの作成**をクリックします。
3. 検索ボックスに「Azure AI Search」と入力し、検索結果から**Azure AI Search**を選択します。
4. **作成**ボタンをクリックします。
5. **基本**タブで以下の情報を入力します：
   - **サブスクリプション**：ご利用の Azure サブスクリプションを選択します。
   - **リソース グループ**：新しいリソースグループを作成するか、既存のものを選択します。
   - **リソース名**：検索サービスの一意の名前を入力します。
   - **リージョン**：ユーザーに最も近いリージョンを選択します。
   - **価格レベル**：要件に合った価格レベルを選択します。テスト用には無料レベルから始めることができます。
6. **確認および作成**をクリックします。
7. 設定内容を確認し、**作成**をクリックして検索サービスを作成します。

## ステップ 2: Azure AI Search の使い始め

1. デプロイが完了したら、Azure ポータルで検索サービスに移動します。
2. 検索サービスの概要ページで、**クイックスタート**ボタンをクリックします。
3. クイックスタートガイドの手順に従い、インデックスの作成、データのアップロード、検索クエリの実行を行います。

### インデックスの作成

1. クイックスタートガイドで、**インデックスの作成**をクリックします。
2. フィールドとその属性（例：データ型、検索可能、フィルター可能）を指定してインデックススキーマを定義します。
3. **作成**をクリックしてインデックスを作成します。

### データのアップロード

1. クイックスタートガイドで、**データのアップロード**をクリックします。
2. データソース（例：Azure Blob Storage、Azure SQL Database）を選択し、必要な接続情報を入力します。
3. データソースのフィールドをインデックスのフィールドにマッピングします。
4. **送信**をクリックしてデータをインデックスにアップロードします。

### 検索クエリの実行

1. クイックスタートガイドで、**Search explorer**をクリックします。
2. 検索ボックスに検索クエリを入力して、検索機能をテストします。
3. 検索結果を確認し、必要に応じてインデックススキーマやデータを調整します。

## ステップ 3: Azure AI Search ツールの活用

Azure AI Search は様々なツールと連携して、検索機能を強化します。高度な設定や操作には、Azure CLI、Python SDK などのツールを利用できます。

### Azure CLI の使用

1. [Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?wt.mc_id=studentamb_258691) の手順に従って Azure CLI をインストールします。
2. 以下のコマンドで Azure CLI にサインインします：
   ```bash
   az login
   ```
3. Azure CLI を使って新しい検索サービスを作成します：
   ```bash
   az search service create --resource-group <resource-group> --name <service-name> --sku Free
   ```
4. Azure CLI を使ってインデックスを作成します：
   ```bash
   az search index create --service-name <service-name> --name <index-name> --fields "field1:type, field2:type"
   ```

### Python SDK の使用

1. Python 用の Azure Cognitive Search クライアントライブラリをインストールします：
   ```bash
   pip install azure-search-documents
   ```
2. 以下の Python コードを使ってインデックスを作成し、ドキュメントをアップロードします：
   ```python
   from azure.core.credentials import AzureKeyCredential
   from azure.search.documents import SearchClient
   from azure.search.documents.indexes import SearchIndexClient
   from azure.search.documents.indexes.models import SearchIndex, SimpleField, edm

   service_endpoint = "https://<service-name>.search.windows.net"
   api_key = "<api-key>"

   index_client = SearchIndexClient(service_endpoint, AzureKeyCredential(api_key))

   fields = [
       SimpleField(name="id", type=edm.String, key=True),
       SimpleField(name="content", type=edm.String, searchable=True),
   ]

   index = SearchIndex(name="sample-index", fields=fields)

   index_client.create_index(index)

   search_client = SearchClient(service_endpoint, "sample-index", AzureKeyCredential(api_key))

   documents = [
       {"id": "1", "content": "Hello world"},
       {"id": "2", "content": "Azure Cognitive Search"}
   ]

   search_client.upload_documents(documents)
   ```

詳細については、以下のドキュメントをご参照ください：

- [Create an Azure Cognitive Search service](https://learn.microsoft.com/en-us/azure/search/search-create-service-portal?wt.mc_id=studentamb_258691)
- [Get started with Azure Cognitive Search](https://learn.microsoft.com/en-us/azure/search/search-get-started-portal?wt.mc_id=studentamb_258691)
- [Azure AI Search Tools](https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/azure-ai-search?tabs=azurecli%2Cpython&pivots=code-examples?wt.mc_id=studentamb_258691)

## まとめ

Azure ポータルと連携ツールを使って、Azure AI Search のセットアップが完了しました。これで、Azure AI Search の高度な機能や能力を活用して、検索ソリューションをさらに強化できます。

さらなるサポートが必要な場合は、[Azure Cognitive Search ドキュメント](https://learn.microsoft.com/en-us/azure/search/?wt.mc_id=studentamb_258691)をご覧ください。

**免責事項**：  
本書類はAI翻訳サービス「[Co-op Translator](https://github.com/Azure/co-op-translator)」を使用して翻訳されています。正確性には努めておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。原文の言語によるオリジナル文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の使用により生じた誤解や誤訳について、一切の責任を負いかねますのでご了承ください。