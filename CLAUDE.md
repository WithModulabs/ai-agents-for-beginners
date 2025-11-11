# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is "AI Agents for Beginners" - an educational course teaching AI agent development through 15+ lessons. The repository is structured as a learning curriculum with each lesson containing documentation, code samples in multiple frameworks, and supporting materials.

**Key Technologies:**
- Python 3.12+ for code samples
- Jupyter Notebooks (.ipynb) for interactive examples
- Multiple AI frameworks: Semantic Kernel, AutoGen, Microsoft Agent Framework (MAF)
- Azure AI services and GitHub Models Marketplace

## Environment Setup

### Initial Configuration

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with appropriate keys
```

### Environment Variables

**For GitHub Models (Free tier):**
- `GITHUB_TOKEN` - Required for GitHub Models Marketplace access
- `GITHUB_MODEL_ID` - Optional, defaults set in notebooks
- `GITHUB_ENDPOINT` - Default: `https://models.github.ai/inference`

**For Azure AI Services (Requires subscription):**
- `AZURE_OPENAI_API_KEY` - Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI endpoint URL
- `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` - Chat model deployment name
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` - Embeddings deployment name
- `PROJECT_ENDPOINT` - Azure AI Foundry project endpoint
- `AZURE_SEARCH_SERVICE_ENDPOINT` - For RAG examples with Azure AI Search
- Additional Azure configuration per `.env.example`

## Repository Structure

```
<lesson-number>-<lesson-name>/
├── README.md                          # Lesson content and theory
├── code_samples/
│   ├── <number>-semantic-kernel.ipynb      # Semantic Kernel + GitHub Models
│   ├── <number>-autogen.ipynb              # AutoGen framework
│   ├── <number>-python-agent-framework.ipynb  # Microsoft Agent Framework (Python)
│   ├── <number>-dotnet-agent-framework.cs  # Microsoft Agent Framework (.NET)
│   └── <number>-azureaiagent.ipynb         # Azure AI Agent Service
└── images/
    └── *.png

translations/<lang-code>/              # Auto-translated content (50+ languages)
translated_images/<lang-code>/         # Localized images
```

## Working with Code Samples

### Framework Selection

Each lesson typically provides examples in multiple frameworks:

1. **Semantic Kernel + GitHub Models** (`*-semantic-kernel.ipynb`)
   - Free tier with GitHub account
   - Good for learning and prototyping
   - Microsoft's orchestration framework

2. **AutoGen** (`*-autogen.ipynb`)
   - Free tier with GitHub Models
   - Multi-agent orchestration
   - Conversational agent patterns

3. **Microsoft Agent Framework** (`*-python-agent-framework.ipynb`, `*-dotnet-agent-framework.cs`)
   - Latest framework from Microsoft
   - Available in Python and .NET variants
   - Production-ready features

4. **Azure AI Agent Service** (`*-azureaiagent.ipynb`)
   - Requires Azure subscription
   - Fully managed service
   - Enterprise features

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to lesson folder, e.g.:
# 04-tool-use/code_samples/04-semantic-kernel.ipynb

# Execute cells sequentially
# Each notebook is self-contained with imports and config
```

### Testing Changes

```bash
# Verify Python environment
python --version  # Should be 3.12+

# Check key packages
pip list | grep -E "(autogen|semantic-kernel|azure-ai|agent-framework)"

# Test notebook execution (validates imports)
jupyter nbconvert --to script <path-to-notebook>.ipynb --stdout | python

# Verify environment variables
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✓' if os.getenv('GITHUB_TOKEN') else '✗ Missing GITHUB_TOKEN')"
```

## Architecture and Design Patterns

### Course Progression

The course is designed to be followed sequentially:
1. **00-course-setup** - Environment and prerequisites
2. **01-intro-to-ai-agents** - AI agent fundamentals
3. **02-explore-agentic-frameworks** - Framework comparison
4. **03-agentic-design-patterns** - Core patterns
5. **04-tool-use** - Function calling and tool integration
6. **05-agentic-rag** - Retrieval-augmented generation
7. **06-building-trustworthy-agents** - Safety and reliability
8. **07-planning-design** - Agent planning capabilities
9. **08-multi-agent** - Multi-agent systems and collaboration
10. **09-metacognition** - Self-reflection and improvement
11. **10-ai-agents-production** - Production deployment
12. **11-agentic-protocols** - MCP, A2A, NLWeb protocols
13. **12-context-engineering** - Context management
14. **13-agent-memory** - Memory systems for agents
15. **14-microsoft-agent-framework** - Deep dive into MAF

### Multi-Language Translation System

- Source files in English at repository root
- Automated translation via GitHub Actions (`.github/workflows/co-op-translator.yml`)
- 50+ languages supported
- Translated content in `translations/<lang-code>/`
- Translated images in `translated_images/<lang-code>/`
- Translation updates automatically on source changes

### Key Dependencies

From `requirements.txt`:
- **Frameworks:** `semantic-kernel`, `autogen-agentchat`, `autogen-core`, `autogen-ext`, `agent-framework`
- **Azure SDK:** `azure-ai-inference`, `azure-ai-projects`, `azure-search-documents`
- **Protocols:** `mcp[cli]` (Model Context Protocol), `a2a-sdk` (Agent-to-Agent)
- **Vector DBs:** `chromadb` (for RAG examples)
- **Utilities:** `python-dotenv`, `pandas`, `numpy`, `pillow`, `httpx`
- **Memory:** `mem0ai` (agent memory management)
- **Jupyter:** `ipykernel` (notebook support)

## Contributing Guidelines

### File Changes

When modifying code:
- **DO NOT commit:** `.env` files, `venv/`, `__pycache__/`, `*.pyc`
- **DO commit:** Notebook outputs when they demonstrate concepts
- **DO remove:** Temporary files, backup notebooks (`*-backup.ipynb`)
- **DO update:** README.md if changing lesson concepts

### Notebook Conventions

- Use markdown cells to explain concepts before code
- Keep execution order linear (cell 1 → 2 → 3...)
- Include example outputs in notebooks for reference
- Use descriptive variable names matching lesson concepts
- Test full notebook execution before committing

### Code Style

- Follow PEP 8 for Python code
- Use clear, educational-focused code (not production-optimized)
- Add comments for complex concepts
- Group imports: standard library, third-party, local
- Keep notebook cells focused and single-purpose

### Pull Request Process

1. **Test notebooks** - Run all cells, verify no errors
2. **Update documentation** - Modify README.md if needed
3. **Maintain consistency** - Follow patterns from other lessons
4. **Use descriptive titles:**
   - `[Lesson-XX] Add example for <concept>`
   - `[Fix] Correct error in lesson-XX`
   - `[Update] Improve code sample in lesson-XX`
   - `[Docs] Update setup instructions`

## Common Development Scenarios

### Adding a New Code Example

1. Determine which framework(s) to use
2. Create notebook in `<lesson>/code_samples/`
3. Follow naming: `<lesson-number>-<framework>.ipynb`
4. Include markdown cells explaining the concept
5. Add imports and environment setup in first cells
6. Test complete execution
7. Update lesson README.md if needed

### Updating Existing Examples

1. Read the lesson README.md to understand context
2. Locate notebook in `code_samples/`
3. Make changes preserving educational clarity
4. Test full notebook execution
5. Verify outputs demonstrate the concept
6. Check if README needs updates

### Working with Translations

- **Never manually edit** files in `translations/` - they're auto-generated
- Edit source files in English at repository root
- GitHub Actions workflow handles translation
- Translation updates occur automatically on merge to main

### Testing Azure AI Services Examples

Some notebooks require Azure subscription:
- Ensure Azure environment variables in `.env`
- Check Azure region availability for services
- Be aware of quota limits and costs
- GitHub Models examples provide free alternative

## Special Notes

### AGENTS.md File

The repository includes an `AGENTS.md` file with detailed setup instructions and architecture documentation. This file is comprehensive and should be referenced for:
- Complete dependency information
- Detailed framework comparisons
- Learning path recommendations
- Troubleshooting common issues

### Development Container

`.devcontainer/` configuration available for VS Code:
- Provides pre-configured Python environment
- Includes all required extensions
- Simplifies setup for contributors

### Community and Support

- **Discord:** [Azure AI Foundry Community Discord](https://aka.ms/ai-agents/discord) - for questions and discussion
- **Issues:** Use GitHub Issues for bugs, errors, or suggestions
- **Discussions:** Check lesson README files for specific guidance
- **Documentation:** Refer to main README.md for course overview

### Git Workflow Notes

- Main branch: `main`
- Repository uses shallow clones recommended for workshops
- Large repository size (~3 GB full history) - consider `--depth 1` for cloning
- Sparse checkout supported for working with specific lessons only
