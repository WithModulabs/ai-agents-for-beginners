<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:15:35+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "pt"
}
-->
# Guia de Integração do MCP Server

## Pré-requisitos
- Node.js instalado (versão 14 ou superior)
- Gerenciador de pacotes npm
- Ambiente Python com as dependências necessárias

## Passos para Configuração

1. **Instalar o Pacote MCP Server**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **Iniciar o MCP Server**
   ```bash
   npx @modelcontextprotocol/server-github
   ```
   O servidor deve iniciar e exibir uma URL de conexão.

3. **Verificar a Conexão**
   - Procure o ícone de plugue (🔌) na sua interface Chainlit
   - Um número (1) deve aparecer ao lado do ícone indicando conexão bem-sucedida
   - O console deve mostrar: "GitHub plugin setup completed successfully" (junto com outras linhas de status)

## Solução de Problemas

### Problemas Comuns

1. **Conflito de Porta**
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```
   Solução: Altere a porta usando:
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **Problemas de Autenticação**
   - Certifique-se de que as credenciais do GitHub estão configuradas corretamente
   - Verifique se o arquivo .env contém os tokens necessários
   - Confirme o acesso à API do GitHub

3. **Falha na Conexão**
   - Confirme que o servidor está rodando na porta esperada
   - Verifique as configurações do firewall
   - Certifique-se de que o ambiente Python tem os pacotes necessários

## Verificação da Conexão

Seu servidor MCP está conectado corretamente quando:
1. O console exibe "GitHub plugin setup completed successfully"
2. Os logs de conexão mostram "✓ MCP Connection Status: Active"
3. Os comandos do GitHub funcionam na interface de chat

## Variáveis de Ambiente

Necessárias no seu arquivo .env:
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## Testando a Conexão

Envie esta mensagem de teste no chat:
```
Show me the repositories for username: [GitHub Username]
```
Uma resposta bem-sucedida mostrará informações do repositório.

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se a tradução profissional feita por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.