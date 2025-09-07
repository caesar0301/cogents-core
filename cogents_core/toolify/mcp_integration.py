"""
MCP (Model Context Protocol) integration for cogents tools.

Provides functionality to run external MCP servers and integrate their tools
into the cogents toolkit system.
"""

from typing import Any, Dict, List, Optional

from cogents_core.utils.logging import get_logger

from .base import AsyncBaseToolkit, MCPNotAvailableError
from .config import ToolkitConfig
from .registry import register_toolkit

try:
    import mcp.client.session as mcp_session
    import mcp.client.stdio as mcp_stdio
    import mcp.types as mcp_types

    MCP_AVAILABLE = True
except ImportError:
    mcp_session = None
    mcp_stdio = None
    mcp_types = None
    MCP_AVAILABLE = False

logger = get_logger(__name__)


@register_toolkit("mcp")
class MCPToolkit(AsyncBaseToolkit):
    """
    Toolkit for integrating external MCP (Model Context Protocol) servers.

    This toolkit allows cogents to communicate with external MCP servers,
    enabling integration with a wide variety of tools and services that
    implement the MCP protocol.

    Configuration:
    - server_path: Path to the MCP server executable
    - server_args: Arguments to pass to the server
    - server_env: Environment variables for the server
    - tools_filter: Optional list of tool names to include (if None, includes all)
    """

    def __init__(self, config: ToolkitConfig = None):
        """
        Initialize the MCP toolkit.

        Args:
            config: Toolkit configuration containing MCP server details
        """
        super().__init__(config)

        if not MCP_AVAILABLE:
            raise MCPNotAvailableError("MCP package is not available. Install with: pip install mcp")

        # MCP server configuration
        self.server_path = self.config.mcp_server_path
        self.server_args = self.config.mcp_server_args or []
        self.server_env = self.config.mcp_server_env or {}

        if not self.server_path:
            raise ValueError("mcp_server_path must be specified in config")

        # MCP client state
        self.session: Optional[mcp_session.ClientSession] = None
        self.stdio_client: Optional[mcp_stdio.StdioServerParameters] = None
        self._available_tools: Dict[str, mcp_types.Tool] = {}

    async def build(self):
        """Initialize the MCP client and connect to the server."""
        await super().build()

        if self.session is None:
            await self._connect_to_server()
            await self._discover_tools()

    async def cleanup(self):
        """Cleanup MCP client resources."""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                self.logger.warning(f"Error closing MCP session: {e}")

        self.session = None
        self.stdio_client = None
        self._available_tools = {}

        await super().cleanup()

    async def _connect_to_server(self):
        """Connect to the MCP server."""
        try:
            self.logger.info(f"Connecting to MCP server: {self.server_path}")

            # Create stdio server parameters
            self.stdio_client = mcp_stdio.StdioServerParameters(
                command=self.server_path, args=self.server_args, env=self.server_env
            )

            # Create and initialize the session
            self.session = await mcp_stdio.stdio_client(self.stdio_client)

            # Initialize the session
            await self.session.initialize()

            self.logger.info("Successfully connected to MCP server")

        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def _discover_tools(self):
        """Discover available tools from the MCP server."""
        try:
            if not self.session:
                raise RuntimeError("MCP session not initialized")

            # List available tools
            tools_response = await self.session.list_tools()

            self._available_tools = {}
            for tool in tools_response.tools:
                # Filter tools if specified in config
                if self.config.activated_tools is None or tool.name in self.config.activated_tools:
                    self._available_tools[tool.name] = tool

            self.logger.info(f"Discovered {len(self._available_tools)} MCP tools: {list(self._available_tools.keys())}")

        except Exception as e:
            self.logger.error(f"Failed to discover MCP tools: {e}")
            raise

    async def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool with the given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError("MCP session not initialized")

        if tool_name not in self._available_tools:
            raise ValueError(f"Tool '{tool_name}' not available. Available tools: {list(self._available_tools.keys())}")

        try:
            self.logger.debug(f"Calling MCP tool '{tool_name}' with arguments: {arguments}")

            # Call the tool
            result = await self.session.call_tool(tool_name, arguments)

            # Extract content from the result
            if hasattr(result, "content") and result.content:
                # Handle different content types
                content_parts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        content_parts.append(content.text)
                    elif hasattr(content, "data"):
                        # Handle binary data
                        content_parts.append(f"[Binary data: {len(content.data)} bytes]")
                    else:
                        content_parts.append(str(content))

                return "\n".join(content_parts) if content_parts else str(result)
            else:
                return str(result)

        except Exception as e:
            self.logger.error(f"MCP tool call failed for '{tool_name}': {e}")
            raise

    async def get_tools_map(self) -> Dict[str, Any]:
        """
        Get the mapping of tool names to their implementation functions.

        This creates wrapper functions for each MCP tool that handle the
        argument passing and result processing.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        tools_map = {}

        for tool_name, tool_info in self._available_tools.items():
            # Create a wrapper function for each MCP tool
            async def tool_wrapper(tool_name=tool_name, **kwargs):
                return await self._call_mcp_tool(tool_name, kwargs)

            # Set function metadata for better introspection
            tool_wrapper.__name__ = tool_name
            tool_wrapper.__doc__ = tool_info.description

            tools_map[tool_name] = tool_wrapper

        return tools_map

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about all available MCP tools.

        Returns:
            List of tool information dictionaries
        """
        tools_info = []
        for tool_name, tool in self._available_tools.items():
            info = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }
            tools_info.append(info)

        return tools_info

    async def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool information dictionary or None if not found
        """
        if tool_name not in self._available_tools:
            return None

        tool = self._available_tools[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
        }


def create_mcp_toolkit(
    server_path: str,
    server_args: Optional[List[str]] = None,
    server_env: Optional[Dict[str, str]] = None,
    activated_tools: Optional[List[str]] = None,
    **config_params,
) -> MCPToolkit:
    """
    Convenience function to create an MCP toolkit.

    Args:
        server_path: Path to the MCP server executable
        server_args: Arguments to pass to the server
        server_env: Environment variables for the server
        activated_tools: List of tools to activate (None for all)
        **config_params: Additional configuration parameters

    Returns:
        Configured MCPToolkit instance
    """
    config = ToolkitConfig(
        mode="mcp",
        name="mcp",
        activated_tools=activated_tools,
        mcp_server_path=server_path,
        mcp_server_args=server_args or [],
        mcp_server_env=server_env or {},
        config=config_params,
    )

    return MCPToolkit(config=config)
