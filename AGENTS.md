# Cogents Development Guide

## Project Overview

Cogents is a collection of essential building blocks for constructing sophisticated multi-agent systems.

### Key Modules
- `cogents_core.llm` - LLM provider abstractions with support for OpenAI, OpenRouter, Ollama, and LlamaCpp
- `cogents_core.base` - Base classes and utilities for agents, search, vector stores, and web surfing
- `cogents_core.goalith` - Goal management and planning system with decomposition, conflict detection, and replanning
- `cogents_core.memory` - Memory management for agents
- `cogents_core.toolify` - Tool integration framework with registry and execution capabilities

### LLM Providers
Supported providers in `cogents_core.llm`:
- `openai` - OpenAI API compatible services
- `openrouter` - OpenRouter API
- `ollama` - Local Ollama instances  
- `llamacpp` - Local inference with llama-cpp-python (requires LLAMACPP_MODEL_PATH)

## Project Structure

```
cogents/
├── core/                    # Core functionality
│   ├── base/               # Base classes and utilities
│   │   ├── llm/           # LLM provider implementations
│   │   ├── msgbus/        # Message bus system
│   │   ├── routing/       # LLM routing strategies
│   │   └── tracing/       # Tracing and monitoring
│   ├── goalith/           # Goal management system
│   │   ├── conflict/      # Conflict detection
│   │   ├── decomposer/    # Goal decomposition strategies
│   │   ├── goalgraph/     # Goal graph data structures
│   │   └── replanner/     # Dynamic replanning
│   ├── memory/            # Memory management (planned)
│   └── toolify/           # Tool integration framework
├── examples/              # Usage examples
│   ├── base/             # Base functionality examples
│   └── goalith/          # Goalith examples
└── tests/                # Test suite
    ├── base/             # Base functionality tests
    ├── goalith/          # Goalith tests
    └── toolify/          # Toolify tests
```

### Core Module Details

**Base (`cogents_core.base`)**
- `base_agent.py` - Abstract base agent class
- `base_search.py` - Search functionality base
- `base_vectorstore.py` - Vector store abstractions
- `base_websurfer.py` - Web surfing capabilities
- `llm/` - LLM provider implementations (OpenAI, OpenRouter, Ollama, LlamaCpp, LiteLLM)
- `msgbus/` - Message bus system for agent communication
- `routing/` - LLM routing strategies (dynamic complexity, self-assessment)
- `tracing/` - Token tracking and Opik tracing integration

**Goalith (`cogents_core.goalith`)**
- `decomposer/` - Goal decomposition strategies (LLM-based, callable, simple)
- `goalgraph/` - Goal graph data structures and operations
- `conflict/` - Goal conflict detection and resolution
- `replanner/` - Dynamic goal replanning capabilities
- `service.py` - Main Goalith service interface

**Toolify (`cogents_core.toolify`)**
- `registry.py` - Tool registration and management
- `base.py` - Base tool classes and interfaces
- `config.py` - Tool configuration management
- `mcp_integration.py` - Model Context Protocol integration

**Memory (`cogents_core.memory`)**
- Memory management system (currently in development)
- Will provide persistent memory capabilities for agents

### Additional Base Components

**Message Bus (`cogents_core.msgbus`)**
- Inter-agent communication system
- Event-driven architecture support
- Message routing and delivery

**Routing (`cogents_core.routing`)**
- LLM routing strategies for optimal model selection
- Dynamic complexity-based routing
- Self-assessment routing for task complexity evaluation

**Tracing (`cogents_core.tracing`)**
- Token usage tracking and monitoring
- Opik tracing integration for observability
- LangGraph hooks for workflow tracing

## Development Workflow

### After Each Implementation
1. **Run unit tests**: `make test-unit` to ensure tests pass
2. **Format code**: `make format` to apply consistent formatting
3. **Check quality**: `make quality` for comprehensive code quality checks

### Quick Development Checks
- `make dev-check` - Quality + unit tests (fast feedback)
- `make full-check` - All checks + tests + build (comprehensive)
- `make ci-test` - CI test suite
- `make ci-quality` - CI quality checks

### Python Command Execution

- **Always use `poetry run` for Python commands in development**
  - Use `poetry run python script.py` instead of `python script.py`
  - Use `poetry run pytest` instead of `pytest`
  - Use `poetry run python -m module` instead of `python -m module`
  - This ensures proper dependency management and virtual environment isolation

## Examples

```bash
# ✅ Correct
poetry run python examples/base/llamacpp_demo.py
poetry run python examples/goalith/goalith_decomposer_example.py
poetry run pytest tests/
poetry run python -m cogents_core.example

# ❌ Incorrect
python examples/base/llamacpp_demo.py
pytest tests/
python -m cogents_core.example
```

## Usage Examples

### Basic Imports

```python
# LLM clients
from cogents_core.llm import get_llm_client

# Goal management
from cogents_core.goalith.service import GoalithService
from cogents_core.goalith.decomposer import LLMDecomposer

# Tool management
from cogents_core.toolify.registry import ToolRegistry

# Base classes
from cogents_core.base.base_agent import BaseAgent
```

### LLM Usage Examples

```python
from cogents_core.llm import get_llm_client

# OpenAI/OpenRouter providers
client = get_llm_client(provider="openai", api_key="sk-...")
client = get_llm_client(provider="openrouter", api_key="sk-...")

# Local providers
client = get_llm_client(provider="ollama", base_url="http://localhost:11434")
client = get_llm_client(provider="llamacpp", model_path="/path/to/model.gguf")

# LlamaCpp with custom settings
client = get_llm_client(
    provider="llamacpp", 
    model_path="/path/to/model.gguf",
    n_ctx=4096,              # Context window size
    n_gpu_layers=32,         # GPU layers (-1 for all)
    instructor=True          # Enable structured output
)

# Basic chat
response = client.completion([
    {"role": "user", "content": "Hello!"}
])

# Structured output (requires instructor=True)
client = get_llm_client(provider="openai", instructor=True)
result = client.structured_completion(messages, MyPydanticModel)
```

### Environment Variables

For llamacpp provider, you can set:
- `LLAMACPP_MODEL_PATH` - Path to your GGUF model file
- `LLAMACPP_CHAT_MODEL` - Model name for logging (optional)

## Testing

- Integration tests are marked as `pytest.mark.integration`
- Use `make test-unit` to run unit tests
- Use `make test-integration` to run integration tests
- Use `make test` to run all tests
- Use `poetry run pytest tests/` to run all tests (manual)
- Use `poetry run pytest -m integration` for integration tests only (manual)
- Use `poetry run pytest -m "not slow"` to skip slow tests (manual)

### Test Classification Rules

**Integration Tests**: Mark tests with `@pytest.mark.integration` if they:
- Depend on external API services (OpenAI, OpenRouter, Gemini, etc.)
- Require network connectivity to third-party services
- Need API keys or authentication tokens
- Make actual HTTP requests to external services

**Unit Tests**: Tests that can run locally without external dependencies:
- Use mocked external services
- Test local functionality only (file operations, data processing, etc.)
- Don't require network connectivity
- Can run in isolated environments

**Examples:**
```python
# Integration test - requires OpenAI API
@pytest.mark.integration
async def test_analyze_image_with_openai(self, image_toolkit):
    result = await image_toolkit.analyze_image("image.jpg", "Describe this")
    assert "description" in result

# Unit test - uses mocks, runs locally
async def test_get_image_info_success(self, image_toolkit):
    with patch.object(image_toolkit, "_load_image") as mock_load:
        mock_image = MagicMock()
        mock_image.size = (800, 600)
        mock_load.return_value = mock_image

        result = await image_toolkit.get_image_info("image.jpg")
        assert result["width"] == 800
```

## Code Quality

- Use `make format` to format code (black, isort, autoflake)
- Use `make format-check` to check formatting without changes
- Use `make lint` to run linting (flake8, mypy)
- Use `make quality` to run all quality checks
- Use `make autofix` to auto-fix code quality issues
- Keep the first-level folder stuctures of `cogents`, `examples`, and `tests` as the same.

## Development Environment Tips

- Use `poetry install` to install dependencies
- Use `poetry shell` to activate the virtual environment
- Use `poetry add <package>` to add new dependencies
- Use `poetry update` to update existing dependencies

## PR Guidelines

- Title format: `[<module_name>] <Description>`
- Always run `make quality` and `make test` before committing
- Ensure all tests pass before submitting PRs
- Update documentation for any new features or breaking changes
- Follow the existing code style and patterns

## Environment Variables Documentation

### Rule: Production-Only Environment Variables
When creating or updating `env.example` files, **ONLY include environment variables that are actually used in production code**.

**What to include:**
- Environment variables used in `cogents/core/**/*.py` files
- Variables that control runtime behavior of the application
- Configuration variables that users need to set for production deployment

**What to exclude:**
- Environment variables only used in `examples/**/*.py` files
- Variables only used in `tests/**/*.py` files
- Variables only used in `thirdparty/**/*.py` files
- Test-specific configuration variables
- Example-specific configuration variables

**How to verify:**
1. Search for `os.getenv` and `os.environ` usage in production code
2. Use pattern: `cogents/core/**/*.py` (exclude examples, tests, thirdparty)
3. Only document variables that appear in production code search results

**Example:**
```bash
# ✅ Include - used in production code
grep_search query="os\.getenv|os\.environ" include_pattern="cogents/core/**/*.py" exclude_pattern="**/tests/**|**/examples/**|**/thirdparty/**"

# ❌ Don't include - only used in examples/tests
grep_search query="os\.getenv|os\.environ" include_pattern="examples/**/*.py"
```

This ensures that `env.example` files are accurate and only show variables that users actually need to configure in production environments.

## Rationale

Using `poetry run` ensures:
- Consistent dependency versions across development environments
- Proper virtual environment activation
- Access to all project dependencies
- Isolation from system Python packages
