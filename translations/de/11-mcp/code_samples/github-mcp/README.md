<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:13:14+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "de"
}
-->
# Github MCP Server Beispiel

## Beschreibung

Dies ist eine Demo, die für den AI Agents Hackathon erstellt wurde, der im Microsoft Reactor stattfand.

Das Tool wird verwendet, um Hackathon-Projekte basierend auf den Github-Repositories eines Nutzers zu empfehlen.  
Dies geschieht durch:

1. **Github Agent** – Verwendet den Github MCP Server, um Repositories und Informationen zu diesen Repositories abzurufen.  
2. **Hackathon Agent** – Nutzt die Daten des Github Agent, um kreative Hackathon-Projektideen basierend auf den Projekten, den vom Nutzer verwendeten Programmiersprachen und den Projektkategorien des AI Agents Hackathons zu entwickeln.  
3. **Events Agent** – Basierend auf den Vorschlägen des Hackathon Agents empfiehlt der Events Agent relevante Veranstaltungen aus der AI Agent Hackathon-Reihe.

## Code ausführen

### Umgebungsvariablen

Diese Demo verwendet Azure Open AI Service, Semantic Kernel, den Github MCP Server und Azure AI Search.

Stelle sicher, dass die entsprechenden Umgebungsvariablen gesetzt sind, um diese Tools nutzen zu können:

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
``` 

## Chainlit Server starten

Um eine Verbindung zum MCP Server herzustellen, nutzt diese Demo Chainlit als Chat-Oberfläche.

Um den Server zu starten, verwende folgenden Befehl im Terminal:

```bash
chainlit run app.py -w
```

Damit sollte dein Chainlit Server auf `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md` gestartet werden.

## Verbindung zum MCP Server herstellen

Um eine Verbindung zum Github MCP Server herzustellen, wähle das „Stecker“-Symbol unterhalb des Chat-Feldes „Type your message here..“:

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

Dort kannst du auf „Connect an MCP“ klicken, um den Befehl zum Verbinden mit dem Github MCP Server hinzuzufügen:

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

Ersetze "[YOUR PERSONAL ACCESS TOKEN]" durch deinen tatsächlichen Personal Access Token.

Nach der Verbindung sollte neben dem Stecker-Symbol eine (1) erscheinen, die bestätigt, dass die Verbindung besteht. Falls nicht, versuche den Chainlit Server mit `chainlit run app.py -w` neu zu starten.

## Demo verwenden

Um den Agenten-Workflow zum Empfehlen von Hackathon-Projekten zu starten, kannst du eine Nachricht wie diese eingeben:

„Recommend hackathon projects for the Github user koreyspace“

Der Router Agent analysiert deine Anfrage und entscheidet, welche Kombination von Agenten (GitHub, Hackathon und Events) am besten geeignet ist, um deine Anfrage zu bearbeiten. Die Agenten arbeiten zusammen, um umfassende Empfehlungen basierend auf der Analyse der Github-Repositories, Projektideen und relevanten Tech-Veranstaltungen zu liefern.

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ursprungssprache gilt als maßgebliche Quelle. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die durch die Verwendung dieser Übersetzung entstehen.