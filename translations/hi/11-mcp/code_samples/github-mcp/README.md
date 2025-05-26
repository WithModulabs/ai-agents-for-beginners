<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:14:23+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "hi"
}
-->
# Github MCP Server उदाहरण

## विवरण

यह डेमो Microsoft Reactor के माध्यम से आयोजित AI Agents Hackathon के लिए बनाया गया था।

यह टूल उपयोगकर्ता के Github रिपॉजिटरी के आधार पर हैकाथॉन प्रोजेक्ट्स की सिफारिश करता है।  
यह इस प्रकार किया जाता है:

1. **Github Agent** - Github MCP Server का उपयोग करके रिपॉजिटरी और उनकी जानकारी प्राप्त करता है।  
2. **Hackathon Agent** - Github Agent से प्राप्त डेटा लेकर उपयोगकर्ता के प्रोजेक्ट्स, उपयोग की गई भाषाओं और AI Agents हैकाथॉन के प्रोजेक्ट ट्रैक्स के आधार पर रचनात्मक हैकाथॉन प्रोजेक्ट आइडियाज प्रस्तुत करता है।  
3. **Events Agent** - हैकाथॉन एजेंट के सुझाव के आधार पर, Events Agent AI Agent Hackathon सीरीज के प्रासंगिक इवेंट्स की सिफारिश करता है।  

## कोड चलाना

### Environment Variables

यह डेमो Azure Open AI Service, Semantic Kernel, Github MCP Server और Azure AI Search का उपयोग करता है।

सुनिश्चित करें कि इन टूल्स का उपयोग करने के लिए आपके पास सही environment variables सेट हैं:

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
```  

## Chainlit Server चलाना

MCP सर्वर से कनेक्ट करने के लिए, यह डेमो Chainlit को चैट इंटरफेस के रूप में उपयोग करता है।

सर्वर चलाने के लिए, अपने टर्मिनल में निम्न कमांड का उपयोग करें:

```bash
chainlit run app.py -w
```

यह आपके Chainlit सर्वर को `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md` पर शुरू कर देगा।  

## MCP Server से कनेक्ट करना

Github MCP Server से कनेक्ट करने के लिए, "Type your message here.." चैट बॉक्स के नीचे "plug" आइकन चुनें:

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

यहाँ से आप "Connect an MCP" पर क्लिक करके Github MCP Server से कनेक्ट करने का कमांड जोड़ सकते हैं:

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

"[YOUR PERSONAL ACCESS TOKEN]" को अपने वास्तविक Personal Access Token से बदलें।  

कनेक्ट होने के बाद, प्लग आइकन के पास (1) दिखाई देगा जो कनेक्शन की पुष्टि करता है। यदि नहीं दिखे, तो `chainlit run app.py -w` के साथ chainlit सर्वर को पुनः शुरू करने का प्रयास करें।  

## डेमो का उपयोग करना

हैकाथॉन प्रोजेक्ट्स की सिफारिश शुरू करने के लिए, आप ऐसा संदेश टाइप कर सकते हैं:

"Recommend hackathon projects for the Github user koreyspace"

Router Agent आपके अनुरोध का विश्लेषण करेगा और तय करेगा कि कौन से एजेंट्स (GitHub, Hackathon, और Events) आपके प्रश्न को संभालने के लिए सबसे उपयुक्त हैं। ये एजेंट्स मिलकर GitHub रिपॉजिटरी विश्लेषण, प्रोजेक्ट विचार और प्रासंगिक तकनीकी इवेंट्स के आधार पर व्यापक सिफारिशें प्रदान करते हैं।

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान रखें कि स्वचालित अनुवादों में त्रुटियाँ या गलतियाँ हो सकती हैं। मूल दस्तावेज़ अपनी मूल भाषा में ही प्रामाणिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सलाह दी जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।