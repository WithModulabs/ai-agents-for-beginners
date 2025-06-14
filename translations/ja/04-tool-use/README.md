<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "88258b03f2893aa2e69eb8fb24baabbc",
  "translation_date": "2025-06-11T04:43:57+00:00",
  "source_file": "04-tool-use/README.md",
  "language_code": "ja"
}
-->
[![How to Design Good AI Agents](../../../04-tool-use/images/lesson-4-thumbnail.png)](https://youtu.be/vieRiPRx-gI?si=cEZ8ApnT6Sus9rhn)

> _(上の画像をクリックすると、このレッスンの動画が視聴できます)_

# ツール利用デザインパターン

ツールは、AIエージェントにより広範な能力を持たせるために興味深いものです。エージェントが実行できるアクションが限られている代わりに、ツールを追加することで、エージェントは多様なアクションを実行できるようになります。本章では、AIエージェントが特定のツールを使って目標を達成する方法を説明するツール利用デザインパターンについて見ていきます。

## はじめに

このレッスンでは、以下の質問に答えることを目指します：

- ツール利用デザインパターンとは何か？
- どのようなユースケースに適用できるのか？
- このデザインパターンを実装するために必要な要素や構成要素は何か？
- 信頼できるAIエージェントを構築するためにツール利用デザインパターンを使う際の特別な注意点は何か？

## 学習目標

このレッスンを終えると、以下のことができるようになります：

- ツール利用デザインパターンとその目的を定義できる
- ツール利用デザインパターンが適用可能なユースケースを特定できる
- デザインパターンを実装するための主要な要素を理解できる
- このデザインパターンを用いたAIエージェントの信頼性を確保するための考慮点を認識できる

## ツール利用デザインパターンとは？

**ツール利用デザインパターン**は、LLMに外部ツールと連携して特定の目標を達成する能力を持たせることに焦点を当てています。ツールとは、エージェントが実行してアクションを行うためのコードです。ツールは、計算機能のような単純な関数であったり、株価の照会や天気予報のようなサードパーティのサービスへのAPI呼び出しであったりします。AIエージェントの文脈では、ツールは**モデルが生成した関数呼び出し**に応じてエージェントが実行するよう設計されています。

## どのようなユースケースに適用できるのか？

AIエージェントはツールを活用して、複雑なタスクの完了、情報の取得、意思決定を行うことができます。ツール利用デザインパターンは、データベースやウェブサービス、コードインタープリターなど外部システムと動的にやり取りする必要があるシナリオでよく使われます。この能力は以下のような多様なユースケースに役立ちます：

- **動的情報取得**：エージェントが外部APIやデータベースに問い合わせて最新のデータを取得する（例：SQLiteデータベースからのデータ分析、株価や天気情報の取得）
- **コード実行と解釈**：エージェントがコードやスクリプトを実行して数学問題を解いたり、レポートを生成したり、シミュレーションを行う
- **ワークフロー自動化**：タスクスケジューラー、メールサービス、データパイプラインなどのツールを統合して繰り返し作業や多段階のワークフローを自動化する
- **カスタマーサポート**：CRMシステムやチケット管理プラットフォーム、ナレッジベースと連携してユーザーの問い合わせを解決する
- **コンテンツ生成・編集**：文法チェッカー、テキスト要約ツール、コンテンツ安全性評価ツールなどを活用してコンテンツ作成を支援する

## ツール利用デザインパターンを実装するために必要な要素／構成要素は？

これらの構成要素によってAIエージェントは多様なタスクを実行できます。ツール利用デザインパターンの実装に必要な主要な要素を見てみましょう：

- **関数／ツールスキーマ**：利用可能なツールの詳細な定義。関数名、目的、必要なパラメーター、期待される出力などを含みます。これらのスキーマにより、LLMは利用可能なツールを理解し、有効なリクエストを構築できます。

- **関数実行ロジック**：ユーザーの意図や会話の文脈に基づいて、いつどのようにツールを呼び出すかを管理します。プランナーやルーティング機構、条件分岐などを含み、ツールの使用を動的に決定します。

- **メッセージ処理システム**：ユーザー入力、LLMの応答、ツール呼び出し、ツールの出力間の会話の流れを管理するコンポーネント。

- **ツール統合フレームワーク**：単純な関数から複雑な外部サービスまで、エージェントとさまざまなツールを接続するインフラストラクチャ。

- **エラー処理＆検証**：ツール実行時の失敗処理、パラメーターの検証、予期しない応答の管理の仕組み。

- **状態管理**：会話の文脈、過去のツール操作、永続的なデータを追跡し、多ターンの対話で一貫性を保つ。

次に、関数／ツール呼び出しについて詳しく見ていきましょう。

### 関数／ツール呼び出し

関数呼び出しは、LLMがツールと連携するための主要な方法です。'Function'と'Tool'はしばしば同義で使われます。なぜなら、関数（再利用可能なコードのブロック）がエージェントがタスクを実行するための'ツール'だからです。関数のコードを呼び出すには、LLMがユーザーの要求を関数の説明と照合する必要があります。そのため、利用可能なすべての関数の説明を含むスキーマがLLMに送られます。LLMはタスクに最適な関数を選び、その名前と引数を返します。選択された関数が呼び出され、その応答がLLMに返され、LLMはその情報を使ってユーザーの要求に応答します。

関数呼び出しを実装するために開発者が必要なものは：

1. 関数呼び出しをサポートするLLMモデル
2. 関数説明を含むスキーマ
3. 説明された各関数のコード

都市の現在時刻を取得する例で説明しましょう：

1. **関数呼び出しをサポートするLLMを初期化する：**

    すべてのモデルが関数呼び出しをサポートしているわけではないので、使用するLLMが対応しているか確認することが重要です。<a href="https://learn.microsoft.com/azure/ai-services/openai/how-to/function-calling" target="_blank">Azure OpenAI</a>は関数呼び出しをサポートしています。まずはAzure OpenAIクライアントを初期化しましょう。

    ```python
    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-05-01-preview"
    )
    ```

1. **関数スキーマを作成する：**

    次に、関数名、関数の説明、関数パラメーターの名前と説明を含むJSONスキーマを定義します。
    このスキーマを先ほど作成したクライアントに渡し、サンフランシスコの時刻を取得するユーザーの要求と一緒に送ります。重要なのは、**ツール呼び出し**が返されることであり、質問の最終的な答えではないことです。先述の通り、LLMはタスクに選んだ関数の名前と渡す引数を返します。

    ```python
    # Function description for the model to read
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            }
        }
    ]
    ```
   
    ```python
  
    # Initial user message
    messages = [{"role": "user", "content": "What's the current time in San Francisco"}] 
  
    # First API call: Ask the model to use the function
      response = client.chat.completions.create(
          model=deployment_name,
          messages=messages,
          tools=tools,
          tool_choice="auto",
      )
  
      # Process the model's response
      response_message = response.choices[0].message
      messages.append(response_message)
  
      print("Model's response:")  

      print(response_message)
  
    ```

    ```bash
    Model's response:
    ChatCompletionMessage(content=None, role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_pOsKdUlqvdyttYB67MOj434b', function=Function(arguments='{"location":"San Francisco"}', name='get_current_time'), type='function')])
    ```
  
1. **タスクを実行するための関数コード：**

    LLMが実行すべき関数を選択したので、そのタスクを実行するコードを実装し実行する必要があります。
    Pythonで現在時刻を取得するコードを実装しましょう。また、response_messageから関数名と引数を抽出して最終結果を得るコードも書きます。

    ```python
      def get_current_time(location):
        """Get the current time for a given location"""
        print(f"get_current_time called with location: {location}")  
        location_lower = location.lower()
        
        for key, timezone in TIMEZONE_DATA.items():
            if key in location_lower:
                print(f"Timezone found for {key}")  
                current_time = datetime.now(ZoneInfo(timezone)).strftime("%I:%M %p")
                return json.dumps({
                    "location": location,
                    "current_time": current_time
                })
      
        print(f"No timezone data found for {location_lower}")  
        return json.dumps({"location": location, "current_time": "unknown"})
    ```

    ```python
     # Handle function calls
      if response_message.tool_calls:
          for tool_call in response_message.tool_calls:
              if tool_call.function.name == "get_current_time":
     
                  function_args = json.loads(tool_call.function.arguments)
     
                  time_response = get_current_time(
                      location=function_args.get("location")
                  )
     
                  messages.append({
                      "tool_call_id": tool_call.id,
                      "role": "tool",
                      "name": "get_current_time",
                      "content": time_response,
                  })
      else:
          print("No tool calls were made by the model.")  
  
      # Second API call: Get the final response from the model
      final_response = client.chat.completions.create(
          model=deployment_name,
          messages=messages,
      )
  
      return final_response.choices[0].message.content
     ```

    ```bash
      get_current_time called with location: San Francisco
      Timezone found for san francisco
      The current time in San Francisco is 09:24 AM.
     ```

関数呼び出しは、多くのエージェントのツール利用デザインの中心ですが、ゼロから実装するのは時に難しいこともあります。
[レッスン2](../../../02-explore-agentic-frameworks)で学んだように、エージェントフレームワークはツール利用を実装するためのビルディングブロックを提供してくれます。

## エージェントフレームワークを使ったツール利用の例

以下は、さまざまなエージェントフレームワークを使ってツール利用デザインパターンを実装する例です：

### Semantic Kernel

<a href="https://learn.microsoft.com/azure/ai-services/agents/overview" target="_blank">Semantic Kernel</a>は、.NET、Python、Javaの開発者向けのオープンソースAIフレームワークで、LLMを使った開発を簡素化します。関数呼び出しを、関数やパラメーターの説明をモデルに自動的に伝える<a href="https://learn.microsoft.com/semantic-kernel/concepts/ai-services/chat-completion/function-calling/?pivots=programming-language-python#1-serializing-the-functions" target="_blank">シリアライズ</a>というプロセスで扱います。また、モデルとコード間の通信も処理します。Semantic Kernelのようなエージェントフレームワークを使う利点の一つは、<a href="https://github.com/microsoft/semantic-kernel/blob/main/python/samples/getting_started_with_agents/openai_assistant/step4_assistant_tool_file_search.py" target="_blank">ファイル検索</a>や<a href="https://github.com/microsoft/semantic-kernel/blob/main/python/samples/getting_started_with_agents/openai_assistant/step3_assistant_tool_code_interpreter.py" target="_blank">コードインタープリター</a>のような既成のツールにアクセスできることです。

以下の図はSemantic Kernelでの関数呼び出しの流れを示しています：

![function calling](../../../04-tool-use/images/functioncalling-diagram.png)

Semantic Kernelでは関数／ツールは<a href="https://learn.microsoft.com/semantic-kernel/concepts/plugins/?pivots=programming-language-python" target="_blank">プラグイン</a>と呼ばれます。`get_current_time` function we saw earlier into a plugin by turning it into a class with the function in it. We can also import the `kernel_function`デコレーターに関数の説明を渡して変換できます。その後、GetCurrentTimePluginを使ってカーネルを作成すると、カーネルが自動的に関数とパラメーターをシリアライズし、LLMに送るスキーマを生成します。

```python
from semantic_kernel.functions import kernel_function

class GetCurrentTimePlugin:
    async def __init__(self, location):
        self.location = location

    @kernel_function(
        description="Get the current time for a given location"
    )
    def get_current_time(location: str = ""):
        ...

```

```python 
from semantic_kernel import Kernel

# Create the kernel
kernel = Kernel()

# Create the plugin
get_current_time_plugin = GetCurrentTimePlugin(location)

# Add the plugin to the kernel
kernel.add_plugin(get_current_time_plugin)
```
  
### Azure AI Agent Service

<a href="https://learn.microsoft.com/azure/ai-services/agents/overview" target="_blank">Azure AI Agent Service</a>は、開発者が基盤となる計算資源やストレージを管理せずに、高品質で拡張可能なAIエージェントを安全に構築、展開、スケールできるよう設計された新しいエージェントフレームワークです。特に企業向けアプリケーションに適しており、エンタープライズグレードのセキュリティを備えたフルマネージドサービスです。

LLM APIを直接使う場合と比べて、Azure AI Agent Serviceには以下のような利点があります：

- 自動ツール呼び出し：ツール呼び出しの解析、ツールの実行、応答処理がサーバー側で行われるため不要
- 安全に管理されたデータ：会話状態を自分で管理する代わりに、必要な情報をすべてスレッドに保存可能
- すぐに使えるツール：Bing、Azure AI Search、Azure Functionsなどのデータソースと連携できるツールが利用可能

Azure AI Agent Serviceで利用可能なツールは、以下の2つのカテゴリに分けられます：

1. 知識ツール：
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/bing-grounding?tabs=python&pivots=overview" target="_blank">Bing検索によるグラウンディング</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/file-search?tabs=python&pivots=overview" target="_blank">ファイル検索</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/azure-ai-search?tabs=azurecli%2Cpython&pivots=overview-azure-ai-search" target="_blank">Azure AI Search</a>

2. アクションツール：
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/function-calling?tabs=python&pivots=overview" target="_blank">関数呼び出し</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/code-interpreter?tabs=python&pivots=overview" target="_blank">コードインタープリター</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/openapi-spec?tabs=python&pivots=overview" target="_blank">OpenAI定義ツール</a>
    - <a href="https://learn.microsoft.com/azure/ai-services/agents/how-to/tools/azure-functions?pivots=overview" target="_blank">Azure Functions</a>

Agent Serviceでは、これらのツールを`toolset`. It also utilizes `threads` which keep track of the history of messages from a particular conversation.

Imagine you are a sales agent at a company called Contoso. You want to develop a conversational agent that can answer questions about your sales data.

The following image illustrates how you could use Azure AI Agent Service to analyze your sales data:

![Agentic Service In Action](../../../04-tool-use/images/agent-service-in-action.jpg)

To use any of these tools with the service we can create a client and define a tool or toolset. To implement this practically we can use the following Python code. The LLM will be able to look at the toolset and decide whether to use the user created function, `fetch_sales_data_using_sqlite_query`のように組み合わせたり、ユーザーのリクエストに応じて既成のコードインタープリターを使ったりできます。

```python 
import os
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from fecth_sales_data_functions import fetch_sales_data_using_sqlite_query # fetch_sales_data_using_sqlite_query function which can be found in a fetch_sales_data_functions.py file.
from azure.ai.projects.models import ToolSet, FunctionTool, CodeInterpreterTool

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)

# Initialize function calling agent with the fetch_sales_data_using_sqlite_query function and adding it to the toolset
fetch_data_function = FunctionTool(fetch_sales_data_using_sqlite_query)
toolset = ToolSet()
toolset.add(fetch_data_function)

# Initialize Code Interpreter tool and adding it to the toolset. 
code_interpreter = code_interpreter = CodeInterpreterTool()
toolset = ToolSet()
toolset.add(code_interpreter)

agent = project_client.agents.create_agent(
    model="gpt-4o-mini", name="my-agent", instructions="You are helpful agent", 
    toolset=toolset
)
```

## 信頼できるAIエージェントを構築するためにツール利用デザインパターンを使う際の特別な注意点は？

LLMが動的に生成するSQLに関する一般的な懸念はセキュリティであり、特にSQLインジェクションやデータベースの削除や改ざんなどの悪意ある操作のリスクがあります。これらの懸念は妥当ですが、適切なデータベースアクセス権限の設定によって効果的に軽減できます。ほとんどのデータベースでは読み取り専用に設定することが一般的です。PostgreSQLやAzure SQLのようなデータベースサービスでは、アプリに読み取り専用（SELECT）ロールを割り当てるべきです。

アプリを安全な環境で実行することも保護を強化します。企業のシナリオでは、運用システムから抽出・変換されたデータが読み取り専用のデータベースやデータウェアハウスに格納され、ユーザーフレンドリーなスキーマで提供されることが多いです。この方法により、データの安全性が確保され、パフォーマンスやアクセス性が最適化され、アプリは制限された読み取り専用アクセスのみを持ちます。

## 追加リソース

- <a href="https://microsoft.github.io/build-your-first-agent-with-azure-ai-agent-service-workshop/" target="_blank">Azure AI Agents Service ワークショップ</a>
- <a href="https://github.com/Azure-Samples/contoso-creative-writer/tree/main/docs/workshop" target="_blank">Contoso Creative Writer マルチエージェントワークショップ</a>
- <a href="https://learn.microsoft.com/semantic-kernel/concepts/ai-services/chat-completion/function-calling/?pivots=programming-language-python#1-serializing-the-functions" target="_blank">Semantic Kernel 関数呼び出しチュートリアル</a>
- <a href="https://github.com/microsoft/semantic-kernel/blob/main/python/samples/getting_started_with_agents/openai_assistant/step3_assistant_tool_code_interpreter.py" target="_blank">Semantic Kernel コードインタープリター</a>
- <a href="https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/components/tools.html" target="_blank">Autogen ツール</a>

## 前のレッ

**免責事項**：  
本書類はAI翻訳サービス「Co-op Translator」（https://github.com/Azure/co-op-translator）を使用して翻訳されました。正確性を期しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。原文の言語で記載されたオリジナルの文書が正式な情報源とみなされます。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じた誤解や誤訳について、当方は一切の責任を負いかねます。