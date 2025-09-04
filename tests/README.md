# Cogents Test Suite

This directory contains the unified test suite for cogents, including both unit and integration tests.

## Test Structure

- `conftest.py` - Unified test configuration and shared fixtures
- `unit/` - Unit tests for individual modules
- `integration/` - Integration tests requiring external services

## Configuration

The unified `conftest.py` provides:

### Environment Variables
Configure external services via environment variables:

```bash
# Vectorstore
export WEAVIATE_URL="http://localhost:8080"
export WEAVIATE_AUTH_SECRET="your-secret"
export PGVECTOR_DB="test_vectorstore"
export PGVECTOR_USER="postgres"
export PGVECTOR_PASSWORD="postgres"
export PGVECTOR_HOST="localhost"
export PGVECTOR_PORT="5432"

# LLM Services
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OLLAMA_BASE_URL="http://localhost:11434"
export LLAMACPP_MODEL_PATH="/path/to/model.gguf"

# Web Search
export TAVILY_API_KEY="your-key"
export GOOGLE_AI_SEARCH_API_KEY="your-key"
export GOOGLE_AI_SEARCH_ENGINE_ID="your-engine-id"
```

### Test Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest -m unit
```

### Integration Tests
```bash
pytest -m integration --runintegration
```

### Specific Test Categories
```bash
# Vectorstore tests
pytest -m vectorstore --runintegration

# LLM tests
pytest -m llm --runintegration

# Slow tests
pytest -m slow --runslow
```

### Test Coverage
```bash
pytest --cov=cogents --cov-report=html
```

## Fixtures

The unified conftest.py provides common fixtures:

- `test_collection_name` - Unique collection names
- `embedding_dims` - Standard embedding dimensions
- `weaviate_config` / `pgvector_config` - Vectorstore configs
- `openai_config` / `ollama_config` / `llama_cpp_config` - LLM configs
- `test_vectors` / `test_payloads` / `test_ids` - Sample test data
- `mock_llm_response` - Mock LLM responses
- `sample_document_chunks` - Sample document chunks

## Adding New Tests

1. **Unit Tests**: Place in `tests/unit/` with appropriate module structure
2. **Integration Tests**: Place in `tests/integration/` with required markers
3. **Fixtures**: Add to `tests/conftest.py` if shared across multiple test files
4. **Configuration**: Use environment variables for external service configs

## Best Practices

- Use appropriate test markers
- Leverage shared fixtures from conftest.py
- Clean up resources in test fixtures
- Use environment variables for configuration
- Write both unit and integration tests for new features