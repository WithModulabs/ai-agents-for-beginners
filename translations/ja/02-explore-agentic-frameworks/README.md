<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "da3523bf8fa456371e21d8d14c67305d",
  "translation_date": "2025-06-11T04:52:05+00:00",
  "source_file": "02-explore-agentic-frameworks/README.md",
  "language_code": "ja"
}
-->
。Wikipediaによると、アクターは「並行計算の基本的な構成要素」であり、受け取ったメッセージに応じて、ローカルでの意思決定、さらなるアクターの生成、メッセージの送信、次に受け取るメッセージへの応答方法の決定が可能です。

**ユースケース**：コード生成の自動化、データ分析タスク、計画や調査機能のためのカスタムエージェント構築。

AutoGenの重要なコアコンセプトは以下の通りです：

- **エージェント**。エージェントは以下を持つソフトウェアエンティティです：
  - **メッセージを介して通信**。これらのメッセージは同期的または非同期的である可能性があります。
  - **自身の状態を維持**し、受信したメッセージによって変更されることがあります。
  - **受信したメッセージや状態の変化に応じてアクションを実行**。これらのアクションはエージェントの状態を変更し、メッセージログの更新、新しいメッセージの送信、コードの実行、API呼び出しなどの外部効果を生み出すことがあります。

  以下はチャット機能を持つ独自のエージェントを作成する短いコードスニペットです：

    ```python
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient


    class MyAssistant(RoutedAgent):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            model_client = OpenAIChatCompletionClient(model="gpt-4o")
            self._delegate = AssistantAgent(name, model_client=model_client)
    
        @message_handler
        async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
            print(f"{self.id.type} received message: {message.content}")
            response = await self._delegate.on_messages(
                [TextMessage(content=message.content, source="user")], ctx.cancellation_token
            )
            print(f"{self.id.type} responded: {response.chat_message.content}")
    ```

    上記のコードでは、`MyAssistant` has been created and inherits from `RoutedAgent`. It has a message handler that prints the content of the message and then sends a response using the `AssistantAgent` delegate. Especially note how we assign to `self._delegate` an instance of `AssistantAgent`はチャット完了を処理できる事前構築されたエージェントです。

    次に、このエージェントタイプをAutoGenに知らせてプログラムを開始します：

    ```python
    
    # main.py
    runtime = SingleThreadedAgentRuntime()
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())

    runtime.start()  # Start processing messages in the background.
    await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_agent", "default"))
    ```

    上記のコードでは、エージェントがランタイムに登録され、メッセージがエージェントに送信され、以下のような出力が得られます：

    ```text
    # Output from the console:
    my_agent received message: Hello, World!
    my_assistant received message: Hello, World!
    my_assistant responded: Hello! How can I assist you today?
    ```

- **マルチエージェント**。AutoGenは複数のエージェントが協力して複雑なタスクを達成することをサポートします。エージェントは通信し、情報を共有し、行動を調整して効率的に問題を解決します。マルチエージェントシステムを作成するには、データ取得、分析、意思決定、ユーザーインタラクションなどの専門機能や役割を持つ異なるタイプのエージェントを定義します。以下にその例を示します：

    ```python
    editor_description = "Editor for planning and reviewing the content."

    # Example of declaring an Agent
    editor_agent_type = await EditorAgent.register(
    runtime,
    editor_topic_type,  # Using topic type as the agent type.
    lambda: EditorAgent(
        description=editor_description,
        group_chat_topic_type=group_chat_topic_type,
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            # api_key="YOUR_API_KEY",
        ),
        ),
    )

    # remaining declarations shortened for brevity

    # Group chat
    group_chat_manager_type = await GroupChatManager.register(
    runtime,
    "group_chat_manager",
    lambda: GroupChatManager(
        participant_topic_types=[writer_topic_type, illustrator_topic_type, editor_topic_type, user_topic_type],
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            # api_key="YOUR_API_KEY",
        ),
        participant_descriptions=[
            writer_description, 
            illustrator_description, 
            editor_description, 
            user_description
        ],
        ),
    )
    ```

    上記コードでは、`GroupChatManager`がランタイムに登録されています。このマネージャーはライター、イラストレーター、編集者、ユーザーなど、異なるタイプのエージェント間のインタラクションを調整する役割を担います。

- **エージェントランタイム**。フレームワークはエージェント間の通信を可能にし、エージェントのIDやライフサイクルを管理し、セキュリティとプライバシーの境界を強制するランタイム環境を提供します。これにより、エージェントを安全かつ制御された環境で実行でき、効率的かつ安全に相互作用できます。注目すべきランタイムは以下の2つです：
  - **スタンドアロンランタイム**。すべてのエージェントが同じプログラミング言語で実装され、同一プロセス内で実行される単一プロセスアプリケーションに適しています。動作イメージは以下の通りです：

アプリケーションスタック

    *エージェントはランタイムを介してメッセージで通信し、ランタイムがエージェントのライフサイクルを管理します*

  - **分散エージェントランタイム**。異なるプログラミング言語で実装され、異なるマシン上で動作する複数プロセスのアプリケーションに適しています。動作イメージは以下の通りです：

## Semantic Kernel + Agent Framework

Semantic Kernelはエンタープライズ対応のAIオーケストレーションSDKです。AIコネクタやメモリコネクタに加え、エージェントフレームワークを含みます。

まず、いくつかのコアコンポーネントを説明します：

- **AIコネクタ**：PythonとC#の両方で使用できる外部AIサービスやデータソースとのインターフェースです。

  ```python
  # Semantic Kernel Python
  from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
  from semantic_kernel.kernel import Kernel

  kernel = Kernel()
  kernel.add_service(
    AzureChatCompletion(
        deployment_name="your-deployment-name",
        api_key="your-api-key",
        endpoint="your-endpoint",
    )
  )
  ```  

    ```csharp
    // Semantic Kernel C#
    using Microsoft.SemanticKernel;

    // Create kernel
    var builder = Kernel.CreateBuilder();
    
    // Add a chat completion service:
    builder.Services.AddAzureOpenAIChatCompletion(
        "your-resource-name",
        "your-endpoint",
        "your-resource-key",
        "deployment-model");
    var kernel = builder.Build();
    ```

    ここでは、カーネルを作成しチャット完了サービスを追加するシンプルな例を示しています。Semantic Kernelは外部AIサービス（この場合はAzure OpenAIチャット完了）への接続を作成します。

- **プラグイン**：アプリケーションで使用可能な機能をカプセル化します。既成のプラグインとカスタムプラグインがあります。関連する概念として「プロンプト関数」があります。自然言語のヒントで関数を呼び出す代わりに、特定の関数をモデルにブロードキャストします。現在のチャットコンテキストに基づき、モデルはこれらの関数のいずれかを呼び出してリクエストやクエリを完了します。例を示します：

  ```python
  from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion


  async def main():
      from semantic_kernel.functions import KernelFunctionFromPrompt
      from semantic_kernel.kernel import Kernel

      kernel = Kernel()
      kernel.add_service(AzureChatCompletion())

      user_input = input("User Input:> ")

      kernel_function = KernelFunctionFromPrompt(
          function_name="SummarizeText",
          prompt="""
          Summarize the provided unstructured text in a sentence that is easy to understand.
          Text to summarize: {{$user_input}}
          """,
      )

      response = await kernel_function.invoke(kernel=kernel, user_input=user_input)
      print(f"Model Response: {response}")

      """
      Sample Console Output:

      User Input:> I like dogs
      Model Response: The text expresses a preference for dogs.
      """


  if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
  ```

    ```csharp
    var userInput = Console.ReadLine();

    // Define semantic function inline.
    string skPrompt = @"Summarize the provided unstructured text in a sentence that is easy to understand.
                        Text to summarize: {{$userInput}}";
    
    // create the function from the prompt
    KernelFunction summarizeFunc = kernel.CreateFunctionFromPrompt(
        promptTemplate: skPrompt,
        functionName: "SummarizeText"
    );

    //then import into the current kernel
    kernel.ImportPluginFromFunctions("SemanticFunctions", [summarizeFunc]);

    ```

    ここでは、テンプレートプロンプト `skPrompt` that leaves room for the user to input text, `$userInput`. Then you create the kernel function `SummarizeText` and then import it into the kernel with the plugin name `SemanticFunctions` を使っています。関数名はSemantic Kernelが関数の内容と呼び出すべきタイミングを理解するのに役立ちます。

- **ネイティブ関数**：フレームワークが直接呼び出してタスクを実行する関数もあります。以下はファイルからコンテンツを取得する例です：

    ```csharp
    public class NativeFunctions {

        [SKFunction, Description("Retrieve content from local file")]
        public async Task<string> RetrieveLocalFile(string fileName, int maxSize = 5000)
        {
            string content = await File.ReadAllTextAsync(fileName);
            if (content.Length <= maxSize) return content;
            return content.Substring(0, maxSize);
        }
    }
    
    //Import native function
    string plugInName = "NativeFunction";
    string functionName = "RetrieveLocalFile";

   //To add the functions to a kernel use the following function
    kernel.ImportPluginFromType<NativeFunctions>();

    ```

- **メモリ**：AIアプリのコンテキスト管理を抽象化し簡素化します。メモリはLLMが知っておくべき情報として扱われます。情報はベクターストア（インメモリデータベースやベクターデータベースなど）に保存できます。以下は非常に簡略化したシナリオで、*facts*をメモリに追加する例です：

    ```csharp
    var facts = new Dictionary<string,string>();
    facts.Add(
        "Azure Machine Learning; https://learn.microsoft.com/azure/machine-learning/",
        @"Azure Machine Learning is a cloud service for accelerating and
        managing the machine learning project lifecycle. Machine learning professionals,
        data scientists, and engineers can use it in their day-to-day workflows"
    );
    
    facts.Add(
        "Azure SQL Service; https://learn.microsoft.com/azure/azure-sql/",
        @"Azure SQL is a family of managed, secure, and intelligent products
        that use the SQL Server database engine in the Azure cloud."
    );
    
    string memoryCollectionName = "SummarizedAzureDocs";
    
    foreach (var fact in facts) {
        await memoryBuilder.SaveReferenceAsync(
            collection: memoryCollectionName,
            description: fact.Key.Split(";")[1].Trim(),
            text: fact.Value,
            externalId: fact.Key.Split(";")[2].Trim(),
            externalSourceName: "Azure Documentation"
        );
    }
    ```

    これらの事実はメモリコレクション `SummarizedAzureDocs` に保存されます。非常に単純な例ですが、LLMが利用するために情報をメモリに保存できることがわかります。

以上がSemantic Kernelフレームワークの基本です。ではエージェントフレームワークはどうでしょうか？

## Azure AI Agent Service

Azure AI Agent ServiceはMicrosoft Ignite 2024で紹介された比較的新しいサービスです。Llama 3、Mistral、CohereなどのオープンソースLLMを直接呼び出せるなど、より柔軟なモデルでAIエージェントの開発と展開を可能にします。

Azure AI Agent Serviceは強力なエンタープライズセキュリティ機構とデータストレージ手段を提供し、エンタープライズアプリケーションに適しています。

AutoGenやSemantic Kernelのようなマルチエージェントオーケストレーションフレームワークと即座に連携できます。

現在パブリックプレビュー中で、PythonとC#でのエージェント構築をサポートしています。

Semantic Kernel Pythonを使ってユーザー定義プラグイン付きのAzure AI Agentを作成する例：

```python
import asyncio
from typing import Annotated

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents import AuthorRole
from semantic_kernel.functions import kernel_function


# Define a sample plugin for the sample
class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        return "$9.99"


async def main() -> None:
    ai_agent_settings = AzureAIAgentSettings.create()

    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(
            credential=creds,
            conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
        ) as client,
    ):
        # Create agent definition
        agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="Host",
            instructions="Answer questions about the menu.",
        )

        # Create the AzureAI Agent using the defined client and agent definition
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
            plugins=[MenuPlugin()],
        )

        # Create a thread to hold the conversation
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread: AzureAIAgentThread | None = None

        user_inputs = [
            "Hello",
            "What is the special soup?",
            "How much does that cost?",
            "Thank you",
        ]

        try:
            for user_input in user_inputs:
                print(f"# User: '{user_input}'")
                # Invoke the agent for the specified thread
                response = await agent.get_response(
                    messages=user_input,
                    thread_id=thread,
                )
                print(f"# {response.name}: {response.content}")
                thread = response.thread
        finally:
            await thread.delete() if thread else None
            await client.agents.delete_agent(agent.id)


if __name__ == "__main__":
    asyncio.run(main())
```

### コアコンセプト

Azure AI Agent Serviceには以下のコアコンセプトがあります：

- **エージェント**。Azure AI Agent ServiceはAzure AI Foundryと統合されています。AI Foundry内で、AIエージェントは「スマート」マイクロサービスとして機能し、質問応答（RAG）、アクション実行、ワークフローの完全自動化を行います。生成AIモデルの力を活用し、実世界のデータソースへのアクセスやインタラクションを可能にするツールと組み合わせています。エージェントの例：

    ```python
    agent = project_client.agents.create_agent(
        model="gpt-4o-mini",
        name="my-agent",
        instructions="You are helpful agent",
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
    )
    ```

    この例では、`gpt-4o-mini`, a name `my-agent`, and instructions `You are helpful agent`モデルを使ってエージェントを作成し、コード解釈タスクを実行するためのツールとリソースを装備しています。

- **スレッドとメッセージ**。スレッドはエージェントとユーザー間の会話やインタラクションを表す重要な概念です。スレッドは会話の進行状況を追跡し、コンテキスト情報を保存し、インタラクションの状態を管理します。スレッドの例：

    ```python
    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content="Could you please create a bar chart for the operating profit using the following data and provide the file to me? Company A: $1.2 million, Company B: $2.5 million, Company C: $3.0 million, Company D: $1.8 million",
    )
    
    # Ask the agent to perform work on the thread
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    
    # Fetch and log all messages to see the agent's response
    messages = project_client.agents.list_messages(thread_id=thread.id)
    print(f"Messages: {messages}")
    ```

    上記コードではスレッドを作成し、その後スレッドにメッセージを送信しています。`create_and_process_run`を呼び出すことでエージェントにスレッド上での作業を依頼し、最後にメッセージを取得してエージェントの応答をログに記録します。メッセージはテキスト、画像、ファイルなど様々なタイプがあり、例えば画像やテキスト応答など、エージェントの作業結果を示します。開発者はこれらの情報を利用して応答をさらに処理したりユーザーに提示できます。

- **他のAIフレームワークとの統合**。Azure AI Agent ServiceはAutoGenやSemantic Kernelなど他のフレームワークと連携可能で、アプリの一部をこれらのフレームワークで構築し、Agent Serviceをオーケストレーターとして使うことや、Agent Serviceのみで全体を構築することもできます。

**ユースケース**：Azure AI Agent Serviceは安全でスケーラブルかつ柔軟なAIエージェント展開を必要とするエンタープライズアプリケーション向けに設計されています。

## これらのフレームワークの違いは？

これらのフレームワークは多くの共通点がありますが、設計、機能、対象ユースケースにおいていくつかの重要な違いがあります：

## AutoGen

AutoGenはMicrosoft ResearchのAI Frontiers Labが開発したオープンソースフレームワークです。イベント駆動型の分散*エージェント*アプリケーションに焦点を当て、複数のLLMやSLM、ツール、高度なマルチエージェント設計パターンを可能にします。

AutoGenはエージェントを中心に構築されており、エージェントは環境を認識し、意思決定を行い、特定の目標を達成するために行動します。エージェントは非同期メッセージで通信し、独立かつ並行して動作できるため、システムのスケーラビリティと応答性が向上します。

Wikipediaによれば、アクターは「並行計算の基本的な構成要素」であり、受け取ったメッセージに応じて、ローカルでの意思決定、さらなるアクターの生成、メッセージの送信、次に受け取るメッセージへの応答方法の決定が可能です。

**ユースケース**：コード生成の自動化、データ分析タスク、計画や調査機能のためのカスタムエージェント構築。

AutoGenの重要なコアコンセプトは以下の通りです：

- **エージェント**。エージェントは以下を持つソフトウェアエンティティです：
  - **メッセージを介して通信**。これらのメッセージは同期的または非同期的である可能性があります。
  - **自身の状態を維持**し、受信したメッセージによって変更されることがあります。
  - **受信したメッセージや状態の変化に応じてアクションを実行**。これらのアクションはエージェントの状態を変更し、メッセージログの更新、新しいメッセージの送信、コードの実行、API呼び出しなどの外部効果を生み出すことがあります。

  以下はチャット機能を持つ独自のエージェントを作成する短いコードスニペットです：

    ```python
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient


    class MyAssistant(RoutedAgent):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            model_client = OpenAIChatCompletionClient(model="gpt-4o")
            self._delegate = AssistantAgent(name, model_client=model_client)
    
        @message_handler
        async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
            print(f"{self.id.type} received message: {message.content}")
            response = await self._delegate.on_messages(
                [TextMessage(content=message.content, source="user")], ctx.cancellation_token
            )
            print(f"{self.id.type} responded: {response.chat_message.content}")
    ```

    上記のコードでは、`MyAssistant` has been created and inherits from `RoutedAgent`. It has a message handler that prints the content of the message and then sends a response using the `AssistantAgent` delegate. Especially note how we assign to `self._delegate` an instance of `AssistantAgent`はチャット完了を処理できる事前構築されたエージェントです。

    次に、このエージェントタイプをAutoGenに知らせてプログラムを開始します：

    ```python
    
    # main.py
    runtime = SingleThreadedAgentRuntime()
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())

    runtime.start()  # Start processing messages in the background.
    await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_agent", "default"))
    ```

    上記のコードでは、エージェントがランタイムに登録され、メッセージがエージェントに送信され、以下のような出力が得られます：

    ```text
    # Output from the console:
    my_agent received message: Hello, World!
    my_assistant received message: Hello, World!
    my_assistant responded: Hello! How can I assist you today?
    ```

- **マルチエージェント**。AutoGenは複数のエージェントが協力して複雑なタスクを達成することをサポートします。エージェントは通信し、情報を共有し、行動を調整して効率的に問題を解決します。マルチエージェントシステムを作成するには、データ取得、分析、意思決定、ユーザーインタラクションなどの専門機能や役割を持つ異なるタイプのエージェントを定義します。以下にその例を示します：

    ```python
    editor_description = "Editor for planning and reviewing the content."

    # Example of declaring an Agent
    editor_agent_type = await EditorAgent.register(
    runtime,
    editor_topic_type,  # Using topic type as the agent type.
    lambda: EditorAgent(
        description=editor_description,
        group_chat_topic_type=group_chat_topic_type,
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            # api_key="YOUR_API_KEY",
        ),
        ),
    )

    # remaining declarations shortened for brevity

    # Group chat
    group_chat_manager_type = await GroupChatManager.register(
    runtime,
    "group_chat_manager",
    lambda: GroupChatManager(
        participant_topic_types=[writer_topic_type, illustrator_topic_type, editor_topic_type, user_topic_type],
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            # api_key="YOUR_API_KEY",
        ),
        participant_descriptions=[
            writer_description, 
            illustrator_description, 
            editor_description, 
            user_description
        ],
        ),
    )
    ```

    上記コードでは、`GroupChatManager`がランタイムに登録されています。このマネージャーはライター、イラストレーター、編集者、ユーザーなど、異なるタイプのエージェント間のインタラクションを調整する役割を担います。

- **エージェントランタイム**。フレームワークはエージェント間の通信を可能にし、エージェントのIDやライフサイクルを管理し、セキュリティとプライバシーの境界を強制するランタイム環境を提供します。これにより、エージェントを安全かつ制御された環境で実行でき、効率的かつ安全に相互作用できます。注目すべきランタイムは以下の2つです：
  - **スタンドアロンランタイム**。すべてのエージェントが同じプログラミング言語で実装され、同一プロセス内で実行される単一プロセスアプリケーションに適しています。動作イメージは以下の通りです：

アプリケーションスタック

    *エージェントはランタイムを介してメッセージで通信し、ランタイムがエージェントのライフサイクルを管理します*

  - **分散エージェントランタイム**。異なるプログラミング言語で実装され、異なるマシン上で動作する複数プロセスのアプリケーションに適しています。動作イメージは以下の通りです：

## Semantic Kernel + Agent Framework

Semantic Kernelはエンタープライズ対応のAIオーケストレーションSDKです。AIコネクタやメモリコネクタに加え、エージェントフレームワークを含みます。

まず、いくつかのコアコンポーネントを説明します：

- **AIコネクタ**：PythonとC#の両方で使用できる外部AIサービスやデータソースとのインターフェースです。

  ```python
  # Semantic Kernel Python
  from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
  from semantic_kernel.kernel import Kernel

  kernel = Kernel()
  kernel.add_service(
    AzureChatCompletion(
        deployment_name="your-deployment-name",
        api_key="your-api-key",
        endpoint="your-endpoint",
    )
  )
  ```  

    ```csharp
    // Semantic Kernel C#
    using Microsoft.SemanticKernel;

    // Create kernel
    var builder = Kernel.CreateBuilder();
    
    // Add a chat completion service:
    builder.Services.AddAzureOpenAIChatCompletion(
        "your-resource-name",
        "your-endpoint",
        "your-resource-key",
        "deployment-model");
    var kernel = builder.Build();
    ```

    ここでは、カーネルを作成しチャット完了サービスを追加するシンプルな例を示しています。Semantic Kernelは外部AIサービス（この場合はAzure OpenAIチャット完了）への接続を作成します。

- **プラグイン**：アプリケーションで使用可能な機能をカプセル化します。既成のプラグインとカスタムプラグインがあります。関連する概念として「プロンプト関数」があります。自然言語のヒントで関数を呼び出す代わりに、特定の関数をモデルにブロードキャストします。現在のチャットコンテキストに基づき、モデルはこれらの関数のいずれかを呼び出してリクエストやクエリを完了します。例を示します：

  ```python
  from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion


  async def main():
      from semantic_kernel.functions import KernelFunctionFromPrompt
      from semantic_kernel.kernel import Kernel

      kernel = Kernel()
      kernel.add_service(AzureChatCompletion())

      user_input = input("User Input:> ")

      kernel_function = KernelFunctionFromPrompt(
          function_name="SummarizeText",
          prompt="""
          Summarize the provided unstructured text in a sentence that is easy to understand.
          Text to summarize: {{$user_input}}
          """,
      )

      response = await kernel_function.invoke(kernel=kernel, user_input=user_input)
      print(f"Model Response: {response}")

      """
      Sample Console Output:

      User Input:> I like dogs
      Model Response: The text expresses a preference for dogs.
      """


  if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
  ```

    ```csharp
    var userInput = Console.ReadLine();

    // Define semantic function inline.
    string skPrompt = @"Summarize the provided unstructured text in a sentence that is easy to understand.
                        Text to summarize: {{$userInput}}";
    
    // create the function from the prompt
    KernelFunction summarizeFunc = kernel.CreateFunctionFromPrompt(
        promptTemplate: skPrompt,
        functionName: "SummarizeText"
    );

    //then import into the current kernel
    kernel.ImportPluginFromFunctions("SemanticFunctions", [summarizeFunc]);

    ```

    ここでは、テンプレートプロンプト `
## 前のレッスン

[AIエージェントとエージェントのユースケース入門](../01-intro-to-ai-agents/README.md)

## 次のレッスン

[エージェント設計パターンの理解](../03-agentic-design-patterns/README.md)

**免責事項**：  
本書類はAI翻訳サービス「[Co-op Translator](https://github.com/Azure/co-op-translator)」を使用して翻訳されました。正確性の向上に努めておりますが、自動翻訳には誤りや不正確な部分が含まれる場合があります。原文の言語によるオリジナル文書が正式な情報源として扱われるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の使用により生じたいかなる誤解や誤訳についても、当方は一切責任を負いません。