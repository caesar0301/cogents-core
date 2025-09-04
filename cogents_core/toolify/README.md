# Cogents Tools System

A unified toolkit system for LLM-based agents with support for LangChain tools, MCP (Model Context Protocol) integration, and comprehensive configuration management.

## Features

- **Unified Configuration**: Single `ToolkitConfig` class for all toolkit types
- **LangChain Integration**: Seamless conversion to LangChain `BaseTool` format
- **MCP Support**: Integration with Model Context Protocol servers
- **Built-in LLM Integration**: Uses cogents LLM system with configurable providers
- **Comprehensive Logging**: Integrated with cogents logging system
- **Async/Sync Support**: Both synchronous and asynchronous toolkit implementations
- **Registry System**: Automatic discovery and registration of toolkits
- **Security Features**: Command filtering, validation, and sandboxing

## Quick Start

### Basic Usage

```python
import asyncio
from cogents_core.toolify import get_toolkit, ToolkitConfig

async def main():
    # Create a toolkit with configuration
    config = ToolkitConfig(
        name="my_python_executor",
        config={
            "default_workdir": "./workspace",
            "default_timeout": 30
        }
    )
    
    # Get the toolkit
    toolkit = get_toolkit("python_executor", config)
    
    # Use the toolkit
    result = await toolkit.call_tool("execute_python_code", code="print('Hello World!')")
    print(result)

asyncio.run(main())
```

### Multiple Toolkits

```python
from cogents_core.toolify import get_toolkits_map, ToolkitConfig

# Configure multiple toolkits
configs = {
    "python_executor": ToolkitConfig(
        config={"default_timeout": 10}
    ),
    "bash": ToolkitConfig(
        config={"workspace_root": "/tmp/workspace"}
    ),
    "search": ToolkitConfig(
        config={
            "SERPER_API_KEY": "your_api_key",
            "JINA_API_KEY": "your_jina_key"
        }
    )
}

# Get all toolkits
toolkits = get_toolkits_map(["python_executor", "bash", "search"], configs)

# Use them
for name, toolkit in toolkits.items():
    print(f"Available toolkit: {name}")
```

## Built-in Toolkits

### Python Executor Toolkit

Execute Python code in a controlled environment with matplotlib support.

```python
toolkit = get_toolkit("python_executor")

result = await toolkit.call_tool("execute_python_code", 
    code="""
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.show()
    
    print("Plot created!")
    """,
    workdir="./plots",
    timeout=30
)
```

**Features:**
- IPython-based execution
- Automatic matplotlib plot saving
- File creation tracking
- ANSI escape sequence cleaning
- Timeout protection
- Comprehensive error handling

### Search Toolkit

Web search and content extraction capabilities.

```python
config = ToolkitConfig(config={
    "SERPER_API_KEY": "your_serper_key",
    "JINA_API_KEY": "your_jina_key"
})

toolkit = get_toolkit("search", config)

# Web search
search_result = await toolkit.call_tool("search_google_api", 
    query="Python asyncio tutorial",
    num_results=5
)

# Web content extraction and Q&A
qa_result = await toolkit.call_tool("web_qa",
    url="https://docs.python.org/3/library/asyncio.html",
    question="What is asyncio?"
)
```

**Features:**
- Google search via Serper API
- Web content extraction via Jina Reader API
- Intelligent content filtering
- LLM-powered Q&A on web content
- Related link extraction

### Bash Toolkit

Execute bash commands in a persistent shell environment.

```python
config = ToolkitConfig(config={
    "workspace_root": "/tmp/workspace",
    "timeout": 60
})

toolkit = get_toolkit("bash", config)

# Execute commands
result = await toolkit.call_tool("run_bash", command="ls -la")
result = await toolkit.call_tool("run_bash", command="cd /tmp && pwd")
```

**Features:**
- Persistent shell session
- Command validation and filtering
- Security restrictions
- ANSI escape sequence cleaning
- Automatic shell recovery
- Workspace isolation

## MCP Integration

The toolkit system supports Model Context Protocol (MCP) servers for external tool integration.

```python
from cogents_core.toolify import create_mcp_toolkit

# Create MCP toolkit
mcp_toolkit = create_mcp_toolkit(
    server_path="/path/to/mcp/server",
    server_args=["--config", "config.json"],
    server_env={"API_KEY": "your_key"},
    activated_tools=["tool1", "tool2"]  # Optional filter
)

# Use MCP toolkit
async with mcp_toolkit:
    # List available tools
    tools_info = await mcp_toolkit.list_available_tools()
    
    # Call MCP tools
    result = await mcp_toolkit.call_tool("mcp_tool_name", 
        arg1="value1", 
        arg2="value2"
    )
```

## Configuration

### ToolkitConfig

The `ToolkitConfig` class provides unified configuration for all toolkit types:

```python
from cogents_core.toolify import ToolkitConfig

config = ToolkitConfig(
    # Core configuration
    mode="builtin",  # or "mcp"
    name="my_toolkit",
    activated_tools=["tool1", "tool2"],  # None for all tools
    config={"api_key": "value"},  # Toolkit-specific config
    
    # LLM integration
    llm_provider="openrouter",  # openrouter, openai, ollama, etc.
    llm_model="gpt-4",
    llm_config={"temperature": 0.1},
    
    # MCP configuration (when mode="mcp")
    mcp_server_path="/path/to/server",
    mcp_server_args=["--arg1", "--arg2"],
    mcp_server_env={"ENV_VAR": "value"},
    
    # Logging
    log_level="INFO",
    enable_tracing=True
)
```

### Environment Variables

The system respects these environment variables:

- `COGENTS_LLM_PROVIDER`: Default LLM provider
- `LOG_LEVEL`: Logging level
- `COGENTS_ENABLE_TRACING`: Enable detailed tracing
- `SERPER_API_KEY`: For search toolkit
- `JINA_API_KEY`: For web content extraction

## LangChain Integration

All toolkits can be converted to LangChain format:

```python
# Get LangChain tools
toolkit = get_toolkit("python_executor")
langchain_tools = toolkit.get_langchain_tools()

# Use with LangChain agents
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(
    llm=your_llm,
    tools=langchain_tools,
    prompt=your_prompt
)
```

## Creating Custom Toolkits

### Synchronous Toolkit

```python
from cogents_core.toolify import BaseToolkit, register_toolkit
from typing import Dict, Callable

@register_toolkit("my_custom")
class MyCustomToolkit(BaseToolkit):
    def get_tools_map(self) -> Dict[str, Callable]:
        return {
            "my_tool": self.my_tool_function
        }
    
    def my_tool_function(self, input_text: str) -> str:
        """My custom tool that processes text."""
        return f"Processed: {input_text}"
```

### Asynchronous Toolkit

```python
from cogents_core.toolify import AsyncBaseToolkit, register_toolkit
from typing import Dict, Callable

@register_toolkit("my_async_custom")
class MyAsyncCustomToolkit(AsyncBaseToolkit):
    async def get_tools_map(self) -> Dict[str, Callable]:
        return {
            "async_tool": self.async_tool_function
        }
    
    async def async_tool_function(self, input_text: str) -> str:
        """My async custom tool."""
        # Perform async operations
        await asyncio.sleep(0.1)
        return f"Async processed: {input_text}"
```

## Registry System

The toolkit registry automatically discovers and manages available toolkits:

```python
from cogents_core.toolify import ToolkitRegistry

# List all registered toolkits
toolkits = ToolkitRegistry.list_toolkits()

# Check if a toolkit is registered
if ToolkitRegistry.is_registered("python_executor"):
    toolkit_class = ToolkitRegistry.get_toolkit_class("python_executor")

# Create toolkit instance
toolkit = ToolkitRegistry.create_toolkit("python_executor", config)
```

## Error Handling

The system provides comprehensive error handling:

```python
from cogents_core.toolify import ToolkitError, MCPNotAvailableError

try:
    result = await toolkit.call_tool("nonexistent_tool")
except ToolkitError as e:
    print(f"Toolkit error: {e}")
except MCPNotAvailableError as e:
    print(f"MCP not available: {e}")
```

## Security Considerations

- **Command Filtering**: Bash toolkit filters dangerous commands
- **Workspace Isolation**: Tools operate in controlled directories
- **Timeout Protection**: All operations have configurable timeouts
- **Input Validation**: Parameters are validated before execution
- **Resource Limits**: Output size and execution time limits

## Dependencies

### Required
- `pydantic ^2.0.0`
- `langchain-core ^0.3.72`
- `aiohttp ^3.9.0`

### Optional
- `mcp ^1.0.0` - For MCP integration
- `matplotlib ^3.8.0` - For Python executor plots
- `pexpect ^4.9.0` - For bash toolkit
- `ipython ^8.18.0` - For enhanced Python execution

## Examples

See `examples/tools_demo.py` for comprehensive usage examples.

## Testing

Run the test suite:

```bash
pytest tests/unit/tools/
```

## Contributing

1. Create custom toolkits by inheriting from `BaseToolkit` or `AsyncBaseToolkit`
2. Use the `@register_toolkit` decorator for automatic registration
3. Follow the established patterns for configuration and error handling
4. Add comprehensive tests for new functionality
5. Update documentation for new features
