<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:16:24+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "pl"
}
-->
# MCP Server Integration Guide

## Wymagania wstępne
- Zainstalowany Node.js (wersja 14 lub wyższa)
- Menedżer pakietów npm
- Środowisko Pythona z wymaganymi zależnościami

## Kroki konfiguracji

1. **Zainstaluj pakiet MCP Server**
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **Uruchom MCP Server**
   ```bash
   npx @modelcontextprotocol/server-github
   ```
   Serwer powinien się uruchomić i wyświetlić URL połączenia.

3. **Zweryfikuj połączenie**
   - Poszukaj ikony wtyczki (🔌) w interfejsie Chainlit
   - Obok ikony wtyczki powinien pojawić się numer (1), co oznacza udane połączenie
   - W konsoli powinno pojawić się: "GitHub plugin setup completed successfully" (wraz z dodatkowymi liniami statusu)

## Rozwiązywanie problemów

### Najczęstsze problemy

1. **Konflikt portów**
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```
   Rozwiązanie: Zmień port używając:
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **Problemy z uwierzytelnianiem**
   - Upewnij się, że dane uwierzytelniające GitHub są poprawnie skonfigurowane
   - Sprawdź, czy plik .env zawiera wymagane tokeny
   - Zweryfikuj dostęp do API GitHub

3. **Nieudane połączenie**
   - Potwierdź, że serwer działa na oczekiwanym porcie
   - Sprawdź ustawienia zapory sieciowej
   - Zweryfikuj, czy środowisko Pythona ma wymagane pakiety

## Weryfikacja połączenia

Twój serwer MCP jest poprawnie połączony, gdy:
1. Konsola pokazuje "GitHub plugin setup completed successfully"
2. Logi połączenia pokazują "✓ MCP Connection Status: Active"
3. Polecenia GitHub działają w interfejsie czatu

## Zmienne środowiskowe

Wymagane w pliku .env:
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## Testowanie połączenia

Wyślij tę wiadomość testową na czacie:
```
Show me the repositories for username: [GitHub Username]
```
Poprawna odpowiedź wyświetli informacje o repozytorium.

**Zastrzeżenie**:  
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczeń AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mimo że dokładamy starań, aby tłumaczenie było jak najbardziej precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w języku źródłowym powinien być uznawany za wiarygodne źródło informacji. W przypadku informacji o kluczowym znaczeniu zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.