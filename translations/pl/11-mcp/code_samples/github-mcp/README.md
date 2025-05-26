<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:13:57+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "pl"
}
-->
# Github MCP Server Przykład

## Opis

To była demonstracja stworzona na AI Agents Hackathon organizowany przez Microsoft Reactor.

Narzędzie służy do rekomendowania projektów hackathonowych na podstawie repozytoriów użytkownika na Githubie.  
Dzieje się to poprzez:

1. **Github Agent** - Korzysta z Github MCP Server, aby pobrać repozytoria i informacje o nich.  
2. **Hackathon Agent** - Przetwarza dane od Github Agenta i generuje kreatywne pomysły na projekty hackathonowe bazując na projektach, językach używanych przez użytkownika oraz ścieżkach projektów dla AI Agents hackathonu.  
3. **Events Agent** - Na podstawie sugestii Hackathon Agenta, Events Agent rekomenduje odpowiednie wydarzenia z serii AI Agent Hackathon.

## Uruchamianie kodu

### Zmienne środowiskowe

Ta demonstracja korzysta z Azure Open AI Service, Semantic Kernel, Github MCP Server oraz Azure AI Search.

Upewnij się, że masz poprawnie ustawione zmienne środowiskowe, aby używać tych narzędzi:

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
```

## Uruchamianie serwera Chainlit

Aby połączyć się z MCP serverem, ta demonstracja używa Chainlit jako interfejsu czatu.

Aby uruchomić serwer, użyj następującego polecenia w terminalu:

```bash
chainlit run app.py -w
```

To powinno uruchomić Twój serwer Chainlit na `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md`.

## Łączenie się z MCP Server

Aby połączyć się z Github MCP Server, wybierz ikonę „wtyczki” pod polem czatu „Type your message here..”:

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

Następnie kliknij „Connect an MCP”, aby dodać polecenie łączenia z Github MCP Server:

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

Zamień "[YOUR PERSONAL ACCESS TOKEN]" na swój rzeczywisty Personal Access Token.

Po połączeniu powinieneś zobaczyć (1) obok ikony wtyczki, co potwierdzi połączenie. Jeśli nie, spróbuj ponownie uruchomić serwer chainlit poleceniem `chainlit run app.py -w`.

## Korzystanie z demonstracji

Aby rozpocząć działanie agenta rekomendującego projekty hackathonowe, możesz wpisać wiadomość taką jak:

"Recommend hackathon projects for the Github user koreyspace"

Router Agent przeanalizuje Twoje zapytanie i zdecyduje, która kombinacja agentów (GitHub, Hackathon i Events) najlepiej poradzi sobie z Twoim zapytaniem. Agenci współpracują, aby zapewnić kompleksowe rekomendacje bazujące na analizie repozytoriów Github, pomysłach na projekty oraz odpowiednich wydarzeniach technologicznych.

**Zastrzeżenie**:  
Niniejszy dokument został przetłumaczony za pomocą automatycznej usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mimo że dążymy do dokładności, prosimy pamiętać, że tłumaczenia automatyczne mogą zawierać błędy lub niedokładności. Oryginalny dokument w języku źródłowym powinien być uznawany za źródło wiarygodne. W przypadku informacji krytycznych zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.