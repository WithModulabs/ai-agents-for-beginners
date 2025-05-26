<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "c4be907703b836d1a1c360db20da4de9",
  "translation_date": "2025-05-21T08:15:00+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/MCP_SETUP.md",
  "language_code": "fr"
}
-->
# Guide d'intégration du serveur MCP

## Prérequis
- Node.js installé (version 14 ou supérieure)
- Gestionnaire de paquets npm
- Environnement Python avec les dépendances requises

## Étapes d'installation

1. **Installer le package MCP Server**  
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

2. **Démarrer le serveur MCP**  
   ```bash
   npx @modelcontextprotocol/server-github
   ```  
   Le serveur devrait démarrer et afficher une URL de connexion.

3. **Vérifier la connexion**  
   - Recherchez l'icône de prise (🔌) dans votre interface Chainlit  
   - Un chiffre (1) doit apparaître à côté de l'icône indiquant une connexion réussie  
   - La console doit afficher : "GitHub plugin setup completed successfully" (avec des lignes de statut supplémentaires)

## Dépannage

### Problèmes courants

1. **Conflit de port**  
   ```bash
   Error: listen EADDRINUSE: address already in use
   ```  
   Solution : Changez le port avec :  
   ```bash
   npx @modelcontextprotocol/server-github --port 3001
   ```

2. **Problèmes d'authentification**  
   - Assurez-vous que les identifiants GitHub sont correctement configurés  
   - Vérifiez que le fichier .env contient les tokens requis  
   - Confirmez l'accès à l'API GitHub

3. **Échec de la connexion**  
   - Vérifiez que le serveur fonctionne sur le port attendu  
   - Contrôlez les paramètres du pare-feu  
   - Assurez-vous que l'environnement Python possède les packages nécessaires

## Vérification de la connexion

Votre serveur MCP est correctement connecté lorsque :  
1. La console affiche "GitHub plugin setup completed successfully"  
2. Les logs de connexion montrent "✓ MCP Connection Status: Active"  
3. Les commandes GitHub fonctionnent dans l'interface de chat

## Variables d'environnement

Requises dans votre fichier .env :  
```
GITHUB_TOKEN=your_github_token
MCP_SERVER_PORT=3000  # Optional, default is 3000
```

## Test de connexion

Envoyez ce message de test dans le chat :  
```
Show me the repositories for username: [GitHub Username]
```  
Une réponse réussie affichera les informations du dépôt.

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant foi. Pour les informations critiques, une traduction professionnelle réalisée par un humain est recommandée. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.