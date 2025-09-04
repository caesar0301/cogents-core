"""
Cogents Tools Module

A unified toolkit system for LLM-based agents with support for:
- LangChain tool integration
- MCP (Model Context Protocol) support
- Unified configuration management
- Built-in logging and LLM integration
"""

from .base import AsyncBaseToolkit, BaseToolkit
from .config import ToolkitConfig
from .registry import ToolkitRegistry, get_toolkit, get_toolkits_map

# Import MCP integration if available
try:
    MCP_AVAILABLE = True
    __all__ = [
        "ToolkitConfig",
        "BaseToolkit",
        "AsyncBaseToolkit",
        "ToolkitRegistry",
        "get_toolkit",
        "get_toolkits_map",
        "MCPToolkit",
        "create_mcp_toolkit",
        "MCP_AVAILABLE",
    ]
except ImportError:
    MCP_AVAILABLE = False
    __all__ = [
        "ToolkitConfig",
        "BaseToolkit",
        "AsyncBaseToolkit",
        "ToolkitRegistry",
        "get_toolkit",
        "get_toolkits_map",
        "MCP_AVAILABLE",
    ]
