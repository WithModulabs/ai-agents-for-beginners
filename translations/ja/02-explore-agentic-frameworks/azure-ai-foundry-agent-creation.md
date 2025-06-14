<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7e92870dc0843e13d4dabc620c09d2d9",
  "translation_date": "2025-06-11T04:51:03+00:00",
  "source_file": "02-explore-agentic-frameworks/azure-ai-foundry-agent-creation.md",
  "language_code": "ja"
}
-->
# Azure AI Agent Service Development

この演習では、[Azure AI Foundry portal](https://ai.azure.com/?WT.mc_id=academic-105485-koreyst)のAzure AI Agentサービスツールを使って、フライト予約用のエージェントを作成します。このエージェントはユーザーと対話し、フライトに関する情報を提供できるようになります。

## 前提条件

この演習を完了するには、以下が必要です：
1. 有効なサブスクリプションを持つAzureアカウント。[無料でアカウントを作成](https://azure.microsoft.com/free/?WT.mc_id=academic-105485-koreyst)できます。
2. Azure AI Foundryハブを作成する権限があるか、すでに作成されていること。
    - ロールがContributorまたはOwnerの場合は、このチュートリアルの手順に従えます。

## Azure AI Foundryハブの作成

> **Note:** Azure AI Foundryは以前Azure AI Studioと呼ばれていました。

1. [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-studio/?WT.mc_id=academic-105485-koreyst)のブログ記事にあるガイドラインに従って、Azure AI Foundryハブを作成します。
2. プロジェクトが作成されたら、表示されるヒントを閉じて、Azure AI Foundryポータルのプロジェクトページを確認します。以下の画像のような画面が表示されるはずです。

    ![Azure AI Foundry Project](../../../02-explore-agentic-frameworks/images/azure-ai-foundry.png)

## モデルのデプロイ

1. プロジェクトの左側ペインの**My assets**セクションで、**Models + endpoints**ページを選択します。
2. **Models + endpoints**ページの**Model deployments**タブで、**+ Deploy model**メニューから**Deploy base model**を選びます。
3. リストから`gpt-4o-mini`モデルを検索し、選択して確定します。

    > **Note**: TPMを減らすことで、使用しているサブスクリプションのクォータを使い過ぎるのを防げます。

    ![Model Deployed](../../../02-explore-agentic-frameworks/images/model-deployment.png)

## エージェントの作成

モデルをデプロイしたので、エージェントを作成できます。エージェントはユーザーと対話できる会話型AIモデルです。

1. プロジェクトの左側ペインの**Build & Customize**セクションで、**Agents**ページを選択します。
2. **+ Create agent**をクリックして新しいエージェントを作成します。**Agent Setup**ダイアログボックスで：
    - エージェント名を入力します（例：`FlightAgent`）。
    - 先ほど作成した`gpt-4o-mini`モデルのデプロイメントが選択されていることを確認します。
    - エージェントに従わせたいプロンプトを**Instructions**に設定します。例は以下の通りです：
    ```
    You are FlightAgent, a virtual assistant specialized in handling flight-related queries. Your role includes assisting users with searching for flights, retrieving flight details, checking seat availability, and providing real-time flight status. Follow the instructions below to ensure clarity and effectiveness in your responses:

    ### Task Instructions:
    1. **Recognizing Intent**:
       - Identify the user's intent based on their request, focusing on one of the following categories:
         - Searching for flights
         - Retrieving flight details using a flight ID
         - Checking seat availability for a specified flight
         - Providing real-time flight status using a flight number
       - If the intent is unclear, politely ask users to clarify or provide more details.
        
    2. **Processing Requests**:
        - Depending on the identified intent, perform the required task:
        - For flight searches: Request details such as origin, destination, departure date, and optionally return date.
        - For flight details: Request a valid flight ID.
        - For seat availability: Request the flight ID and date and validate inputs.
        - For flight status: Request a valid flight number.
        - Perform validations on provided data (e.g., formats of dates, flight numbers, or IDs). If the information is incomplete or invalid, return a friendly request for clarification.

    3. **Generating Responses**:
    - Use a tone that is friendly, concise, and supportive.
    - Provide clear and actionable suggestions based on the output of each task.
    - If no data is found or an error occurs, explain it to the user gently and offer alternative actions (e.g., refine search, try another query).
    
    ```
> [!NOTE]
> 詳細なプロンプトについては、[こちらのリポジトリ](https://github.com/ShivamGoyal03/RoamMind)を参照してください。
    
> また、**Knowledge Base**や**Actions**を追加することで、エージェントの機能を拡張し、ユーザーのリクエストに基づいたより多くの情報提供や自動化タスクの実行が可能になります。この演習ではこれらのステップは省略できます。
    
![Agent Setup](../../../02-explore-agentic-frameworks/images/agent-setup.png)

3. 新しいマルチAIエージェントを作成するには、単に**New Agent**をクリックします。作成されたエージェントはAgentsページに表示されます。

## エージェントのテスト

エージェントを作成したら、Azure AI Foundryポータルのプレイグラウンドでユーザーの問い合わせに対する応答をテストできます。

1. エージェントの**Setup**ペイン上部で、**Try in playground**を選択します。
2. **Playground**ペインでチャットウィンドウに問い合わせを入力してエージェントと対話します。例として、「28日にシアトルからニューヨークへのフライトを検索して」と尋ねることができます。

    > **Note**: この演習ではリアルタイムデータを使用していないため、エージェントの応答が正確でない場合があります。目的は、エージェントが指示に基づいてユーザーの問い合わせを理解し応答できるかを確認することです。

    ![Agent Playground](../../../02-explore-agentic-frameworks/images/agent-playground.png)

3. テスト後、さらに多くのインテントやトレーニングデータ、アクションを追加してエージェントの機能を強化できます。

## リソースのクリーンアップ

エージェントのテストが終わったら、追加コストを避けるためにエージェントを削除できます。
1. [Azure portal](https://portal.azure.com)を開き、この演習で使用したハブリソースが配置されているリソースグループの内容を表示します。
2. ツールバーで**Delete resource group**を選択します。
3. リソースグループ名を入力し、削除を確認します。

## リソース

- [Azure AI Foundry documentation](https://learn.microsoft.com/en-us/azure/ai-studio/?WT.mc_id=academic-105485-koreyst)
- [Azure AI Foundry portal](https://ai.azure.com/?WT.mc_id=academic-105485-koreyst)
- [Getting Started with Azure AI Studio](https://techcommunity.microsoft.com/blog/educatordeveloperblog/getting-started-with-azure-ai-studio/4095602?WT.mc_id=academic-105485-koreyst)
- [Fundamentals of AI agents on Azure](https://learn.microsoft.com/en-us/training/modules/ai-agent-fundamentals/?WT.mc_id=academic-105485-koreyst)
- [Azure AI Discord](https://aka.ms/AzureAI/Discord)

**免責事項**：  
本書類はAI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性には努めておりますが、自動翻訳には誤りや不正確な箇所が含まれる可能性があることをご承知ください。原文はあくまで正式な情報源とみなしてください。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の使用により生じたいかなる誤解や解釈違いについても、当方は一切の責任を負いかねます。