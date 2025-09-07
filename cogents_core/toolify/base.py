"""
Base toolkit classes for cogents tools system.

Provides abstract base classes for both synchronous and asynchronous toolkits
with support for LangChain tools, MCP integration, and unified configuration.
"""

import abc
import asyncio
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.tools import BaseTool, tool

from cogents_core.llm import BaseLLMClient
from cogents_core.utils.logging import get_logger

from .config import ToolkitConfig

try:
    import mcp.types as mcp_types

    MCP_AVAILABLE = True
except ImportError:
    mcp_types = None
    MCP_AVAILABLE = False


class ToolkitError(Exception):
    """Base exception for toolkit-related errors."""


class MCPNotAvailableError(ToolkitError):
    """Raised when MCP functionality is requested but not available."""


class ToolConverter:
    """Utility class for converting between different tool formats."""

    @staticmethod
    def langchain_to_mcp(langchain_tool: BaseTool) -> "mcp_types.Tool":
        """
        Convert a LangChain tool to MCP format.

        Args:
            langchain_tool: LangChain BaseTool instance

        Returns:
            MCP Tool instance

        Raises:
            MCPNotAvailableError: If MCP is not available
        """
        if not MCP_AVAILABLE:
            raise MCPNotAvailableError("MCP package is not installed")

        return mcp_types.Tool(
            name=langchain_tool.name,
            description=langchain_tool.description,
            inputSchema=langchain_tool.args_schema.model_json_schema() if langchain_tool.args_schema else {},
        )

    @staticmethod
    def function_to_langchain(
        func: Callable, name: Optional[str] = None, description: Optional[str] = None
    ) -> BaseTool:
        """
        Convert a function to LangChain tool format.

        Args:
            func: Function to convert
            name: Optional tool name (defaults to function name)
            description: Optional tool description (defaults to function docstring)

        Returns:
            LangChain BaseTool instance
        """
        # Use function name if no name provided
        tool_name = name if name is not None else func.__name__

        # Use function docstring if no description provided
        if description is not None:
            tool_description = description
        else:
            # Use docstring and clean it up (add period if needed)
            tool_description = func.__doc__ or ""
            if tool_description and not tool_description.endswith("."):
                tool_description = tool_description.strip() + "."

        return tool(tool_name, description=tool_description)(func)


class BaseToolkit(abc.ABC):
    """
    Base class for synchronous toolkits.

    Provides a common interface for all toolkit implementations with support
    for LangChain tools, configuration management, and logging.
    """

    def __init__(self, config: Optional[Union[ToolkitConfig, Dict[str, Any]]] = None):
        """
        Initialize the toolkit.

        Args:
            config: Toolkit configuration (ToolkitConfig instance or dict)
        """
        if isinstance(config, dict):
            config = ToolkitConfig(**config)
        elif config is None:
            config = ToolkitConfig()

        self.config = config
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._tools_cache: Optional[Dict[str, Callable]] = None
        self._llm_client: Optional[BaseLLMClient] = None

    @property
    def llm_client(self) -> BaseLLMClient:
        """Get or create LLM client instance."""
        if self._llm_client is None:
            self._llm_client = self.config.get_llm_client()
        return self._llm_client

    @abc.abstractmethod
    def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get a mapping of tool names to their implementation functions.

        Returns:
            Dict mapping tool names to callable functions
        """

    def get_filtered_tools_map(self) -> Dict[str, Callable]:
        """
        Get tools map filtered by activated_tools configuration.

        Returns:
            Dict containing only activated tools
        """
        if self._tools_cache is None:
            self._tools_cache = self.get_tools_map()

        if self.config.activated_tools is None:
            return self._tools_cache

        # Validate that all activated tools exist
        missing_tools = set(self.config.activated_tools) - set(self._tools_cache.keys())
        if missing_tools:
            raise ToolkitError(
                f"Activated tools not found in {self.__class__.__name__}: {missing_tools}. "
                f"Available tools: {list(self._tools_cache.keys())}"
            )

        return {name: self._tools_cache[name] for name in self.config.activated_tools}

    def get_langchain_tools(self) -> List[BaseTool]:
        """
        Get tools in LangChain format.

        Returns:
            List of LangChain BaseTool instances
        """
        tools_map = self.get_filtered_tools_map()
        return [ToolConverter.function_to_langchain(func, name) for name, func in tools_map.items()]

    def get_mcp_tools(self) -> List["mcp_types.Tool"]:
        """
        Get tools in MCP format.

        Returns:
            List of MCP Tool instances

        Raises:
            MCPNotAvailableError: If MCP is not available
        """
        if not MCP_AVAILABLE:
            raise MCPNotAvailableError("MCP package is not installed")

        langchain_tools = self.get_langchain_tools()
        return [ToolConverter.langchain_to_mcp(tool) for tool in langchain_tools]

    def call_tool(self, name: str, **kwargs) -> Any:
        """
        Call a tool by name with the provided arguments.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ToolkitError: If tool is not found
        """
        tools_map = self.get_filtered_tools_map()
        if name not in tools_map:
            raise ToolkitError(f"Tool '{name}' not found in {self.__class__.__name__}")

        tool_func = tools_map[name]
        self.logger.debug(f"Calling tool '{name}' with args: {kwargs}")

        try:
            result = tool_func(**kwargs)
            self.logger.debug(f"Tool '{name}' completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Tool '{name}' failed: {e}")
            raise


class AsyncBaseToolkit(BaseToolkit):
    """
    Base class for asynchronous toolkits.

    Extends BaseToolkit with async support and lifecycle management.
    """

    def __init__(self, config: Optional[Union[ToolkitConfig, Dict[str, Any]]] = None):
        """
        Initialize the async toolkit.

        Args:
            config: Toolkit configuration (ToolkitConfig instance or dict)
        """
        super().__init__(config)
        self._built = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.build()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def build(self):
        """
        Build/initialize the toolkit.

        Override this method to perform async initialization tasks.
        """
        if self._built:
            return
        self.logger.debug(f"Building {self.__class__.__name__}")
        self._built = True

    async def cleanup(self):
        """
        Cleanup toolkit resources.

        Override this method to perform cleanup tasks.
        """
        self.logger.debug(f"Cleaning up {self.__class__.__name__}")
        self._built = False

    @abc.abstractmethod
    async def get_tools_map(self) -> Dict[str, Callable]:
        """
        Get a mapping of tool names to their implementation functions.

        Returns:
            Dict mapping tool names to callable functions
        """

    async def get_filtered_tools_map(self) -> Dict[str, Callable]:
        """
        Get tools map filtered by activated_tools configuration.

        Returns:
            Dict containing only activated tools
        """
        if self._tools_cache is None:
            self._tools_cache = await self.get_tools_map()

        if self.config.activated_tools is None:
            return self._tools_cache

        # Validate that all activated tools exist
        missing_tools = set(self.config.activated_tools) - set(self._tools_cache.keys())
        if missing_tools:
            raise ToolkitError(
                f"Activated tools not found in {self.__class__.__name__}: {missing_tools}. "
                f"Available tools: {list(self._tools_cache.keys())}"
            )

        return {name: self._tools_cache[name] for name in self.config.activated_tools}

    async def get_langchain_tools(self) -> List[BaseTool]:
        """
        Get tools in LangChain format.

        Returns:
            List of LangChain BaseTool instances
        """
        tools_map = await self.get_filtered_tools_map()
        return [ToolConverter.function_to_langchain(func, name) for name, func in tools_map.items()]

    async def get_mcp_tools(self) -> List["mcp_types.Tool"]:
        """
        Get tools in MCP format.

        Returns:
            List of MCP Tool instances

        Raises:
            MCPNotAvailableError: If MCP is not available
        """
        if not MCP_AVAILABLE:
            raise MCPNotAvailableError("MCP package is not installed")

        langchain_tools = await self.get_langchain_tools()
        return [ToolConverter.langchain_to_mcp(tool) for tool in langchain_tools]

    async def call_tool(self, name: str, **kwargs) -> Any:
        """
        Call a tool by name with the provided arguments.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ToolkitError: If tool is not found
        """
        tools_map = await self.get_filtered_tools_map()
        if name not in tools_map:
            raise ToolkitError(f"Tool '{name}' not found in {self.__class__.__name__}")

        tool_func = tools_map[name]
        self.logger.debug(f"Calling tool '{name}' with args: {kwargs}")

        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**kwargs)
            else:
                result = tool_func(**kwargs)
            self.logger.debug(f"Tool '{name}' completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Tool '{name}' failed: {e}")
            raise

    # Sync compatibility methods
    def get_tools_map_sync(self) -> Dict[str, Callable]:
        """Synchronous version of get_tools_map for compatibility."""
        loop = asyncio.get_event_loop()
        if not self._built:
            loop.run_until_complete(self.build())
        return loop.run_until_complete(self.get_tools_map())

    def get_langchain_tools_sync(self) -> List[BaseTool]:
        """Synchronous version of get_langchain_tools for compatibility."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.get_langchain_tools())
