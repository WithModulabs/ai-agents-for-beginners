<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:12:32+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "pt"
}
-->
# Exemplo de Servidor Github MCP

## Descrição

Esta foi uma demonstração criada para o AI Agents Hackathon realizado pelo Microsoft Reactor.

A ferramenta é usada para recomendar projetos de hackathon com base nos repositórios Github de um usuário.  
Isso é feito por meio de:

1. **Github Agent** - Utiliza o Github MCP Server para recuperar repositórios e informações sobre esses repositórios.  
2. **Hackathon Agent** - Recebe os dados do Github Agent e cria ideias criativas de projetos para hackathon com base nos projetos, linguagens usadas pelo usuário e nas trilhas do AI Agents hackathon.  
3. **Events Agent** - Com base na sugestão do Hackathon Agent, o Events Agent recomenda eventos relevantes da série AI Agent Hackathon.

## Executando o código

### Variáveis de Ambiente

Esta demonstração utiliza o Azure Open AI Service, Semantic Kernel, o Github MCP Server e Azure AI Search.

Certifique-se de que as variáveis de ambiente corretas estejam configuradas para usar essas ferramentas:

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
```

## Executando o Chainlit Server

Para se conectar ao MCP server, esta demo usa o Chainlit como interface de chat.

Para rodar o servidor, use o seguinte comando no seu terminal:

```bash
chainlit run app.py -w
```

Isso deve iniciar seu Chainlit server em `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md`.

## Conectando ao MCP Server

Para se conectar ao Github MCP Server, selecione o ícone de "plug" abaixo da caixa de chat "Type your message here..":

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

A partir daí, clique em "Connect an MCP" para adicionar o comando que conecta ao Github MCP Server:

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

Substitua "[YOUR PERSONAL ACCESS TOKEN]" pelo seu Token de Acesso Pessoal real.

Após conectar, você deve ver um (1) ao lado do ícone do plug para confirmar que está conectado. Caso contrário, tente reiniciar o servidor chainlit com `chainlit run app.py -w`.

## Usando a Demo

Para iniciar o fluxo de trabalho do agente recomendando projetos para hackathon, você pode digitar uma mensagem como:

"Recommend hackathon projects for the Github user koreyspace"

O Router Agent vai analisar sua solicitação e determinar qual combinação de agentes (GitHub, Hackathon e Events) é mais adequada para responder sua consulta. Os agentes trabalham juntos para fornecer recomendações completas baseadas na análise dos repositórios do Github, ideação de projetos e eventos tecnológicos relevantes.

**Aviso Legal**:  
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se a tradução profissional feita por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.