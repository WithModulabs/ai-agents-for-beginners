<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:12:12+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "ko"
}
-->
# Github MCP Server 예제

## 설명

이 데모는 Microsoft Reactor에서 주최한 AI Agents 해커톤을 위해 만들어졌습니다.

이 도구는 사용자의 Github 저장소를 기반으로 해커톤 프로젝트를 추천하는 데 사용됩니다.  
이는 다음과 같이 이루어집니다:

1. **Github Agent** - Github MCP Server를 사용하여 저장소와 해당 저장소에 대한 정보를 가져옵니다.  
2. **Hackathon Agent** - Github Agent로부터 받은 데이터를 바탕으로, 사용자가 사용하는 언어와 AI Agents 해커톤의 프로젝트 트랙을 고려해 창의적인 해커톤 프로젝트 아이디어를 제안합니다.  
3. **Events Agent** - Hackathon Agent의 제안을 기반으로 AI Agent 해커톤 시리즈의 관련 이벤트를 추천합니다.

## 코드 실행하기

### 환경 변수

이 데모는 Azure Open AI Service, Semantic Kernel, Github MCP Server, Azure AI Search를 사용합니다.

이 도구들을 사용하려면 적절한 환경 변수가 설정되어 있어야 합니다:

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
``` 

## Chainlit 서버 실행하기

MCP 서버에 연결하기 위해 이 데모는 Chainlit을 채팅 인터페이스로 사용합니다.

서버를 실행하려면 터미널에서 다음 명령어를 사용하세요:

```bash
chainlit run app.py -w
```

이 명령어는 `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md`에서 Chainlit 서버를 시작합니다.

## MCP 서버 연결하기

Github MCP Server에 연결하려면 "Type your message here.." 채팅 박스 아래에 있는 "플러그" 아이콘을 선택하세요:

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

그 다음 "Connect an MCP"를 클릭하여 Github MCP Server에 연결하는 명령어를 추가할 수 있습니다:

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

"[YOUR PERSONAL ACCESS TOKEN]"을 실제 개인 액세스 토큰으로 교체하세요.

연결되면 플러그 아이콘 옆에 (1)이 표시되어 연결되었음을 확인할 수 있습니다.  
만약 표시되지 않는다면 `chainlit run app.py -w` 명령어로 Chainlit 서버를 재시작해 보세요.

## 데모 사용법

해커톤 프로젝트 추천 에이전트 워크플로우를 시작하려면 다음과 같이 메시지를 입력할 수 있습니다:

"Recommend hackathon projects for the Github user koreyspace"

Router Agent가 요청을 분석하여 어떤 에이전트 조합(GitHub, Hackathon, Events)이 쿼리를 처리하는 데 가장 적합한지 결정합니다.  
에이전트들은 함께 협력하여 Github 저장소 분석, 프로젝트 아이디어 제안, 관련 기술 이벤트 추천을 기반으로 종합적인 추천을 제공합니다.

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있음을 유의해 주시기 바랍니다. 원본 문서는 해당 언어의 원문이 권위 있는 자료로 간주되어야 합니다. 중요한 정보의 경우 전문적인 인간 번역을 권장합니다. 본 번역의 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.