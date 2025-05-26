<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9bf0395cbc541ce8db2a9699c8678dfc",
  "translation_date": "2025-05-21T08:13:36+00:00",
  "source_file": "11-mcp/code_samples/github-mcp/README.md",
  "language_code": "fa"
}
-->
# مثال سرور Github MCP

## توضیحات

این یک دمو است که برای هکاتون AI Agents که توسط Microsoft Reactor برگزار شده بود، ساخته شده است.

این ابزار برای پیشنهاد پروژه‌های هکاتون بر اساس مخازن Github یک کاربر استفاده می‌شود.  
این کار به صورت زیر انجام می‌شود:

1. **Github Agent** - با استفاده از سرور Github MCP، مخازن و اطلاعات مربوط به آن‌ها را دریافت می‌کند.  
2. **Hackathon Agent** - داده‌های دریافتی از Github Agent را گرفته و ایده‌های خلاقانه برای پروژه‌های هکاتون بر اساس پروژه‌ها، زبان‌های برنامه‌نویسی کاربر و مسیرهای پروژه در هکاتون AI Agents ارائه می‌دهد.  
3. **Events Agent** - بر اساس پیشنهادات Hackathon Agent، این عامل رویدادهای مرتبط با سری هکاتون AI Agent را توصیه می‌کند.  

## اجرای کد

### متغیرهای محیطی

این دمو از Azure Open AI Service، Semantic Kernel، سرور Github MCP و Azure AI Search استفاده می‌کند.

اطمینان حاصل کنید که متغیرهای محیطی مناسب برای استفاده از این ابزارها تنظیم شده‌اند:

```python
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=""
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_API_VERSION=""
AZURE_SEARCH_SERVICE_ENDPOINT=""
AZURE_SEARCH_API_KEY=""
```

## اجرای سرور Chainlit

برای اتصال به سرور MCP، این دمو از Chainlit به عنوان رابط چت استفاده می‌کند.

برای اجرای سرور، از دستور زیر در ترمینال خود استفاده کنید:

```bash
chainlit run app.py -w
```

این دستور سرور Chainlit شما را روی `localhost:8000` as well as populate your Azure AI Search Index with the `event-descriptions.md` راه‌اندازی می‌کند.

## اتصال به سرور MCP

برای اتصال به سرور Github MCP، روی آیکون «پریز» زیر کادر چت "Type your message here.." کلیک کنید:

![MCP Connect](../../../../../11-mcp/code_samples/github-mcp/images/mcp-chainlit-1.png)

سپس می‌توانید روی "Connect an MCP" کلیک کنید تا دستور اتصال به سرور Github MCP اضافه شود:

```bash
npx -y @modelcontextprotocol/server-github --env GITHUB_PERSONAL_ACCESS_TOKEN=[YOUR PERSONAL ACCESS TOKEN]
```

"[YOUR PERSONAL ACCESS TOKEN]" را با توکن دسترسی شخصی واقعی خود جایگزین کنید.

بعد از اتصال، باید عدد (1) کنار آیکون پریز ظاهر شود تا تایید کند اتصال برقرار شده است. اگر نه، سعی کنید سرور chainlit را با دستور `chainlit run app.py -w` ریستارت کنید.

## استفاده از دمو

برای شروع روند کاری عامل در پیشنهاد پروژه‌های هکاتون، می‌توانید پیامی مانند زیر تایپ کنید:

"Recommend hackathon projects for the Github user koreyspace"

عامل Router درخواست شما را تحلیل می‌کند و تعیین می‌کند کدام ترکیب از عوامل (GitHub، Hackathon و Events) بهترین پاسخ را به سوال شما می‌دهد. این عوامل با هم همکاری می‌کنند تا بر اساس تحلیل مخازن Github، ایده‌پردازی پروژه و رویدادهای مرتبط فناوری، پیشنهادات جامعی ارائه دهند.

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما در تلاش برای دقت هستیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است حاوی خطاها یا نادرستی‌هایی باشند. سند اصلی به زبان بومی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حیاتی، توصیه می‌شود از ترجمه حرفه‌ای انسانی استفاده شود. ما مسئول هیچ‌گونه سوءتفاهم یا برداشت نادرستی که ناشی از استفاده از این ترجمه باشد، نیستیم.