"""
Unified configuration system for cogents tools.
"""

import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from cogents_core.llm import get_llm_client


class ToolkitConfig(BaseModel):
    """
    Unified configuration for cogents toolkits.

    This configuration system supports both built-in tools and MCP integration,
    providing a consistent interface for all toolkit types.
    """

    # Core configuration
    mode: Literal["builtin", "mcp"] = "builtin"
    """Toolkit mode: 'builtin' for native tools, 'mcp' for Model Context Protocol tools"""

    name: Optional[str] = None
    """Toolkit name for identification and logging"""

    activated_tools: Optional[List[str]] = None
    """List of specific tools to activate. If None, all tools are activated."""

    config: Dict[str, Any] = Field(default_factory=dict)
    """Toolkit-specific configuration parameters"""

    # LLM Integration
    llm_provider: str = Field(default_factory=lambda: os.getenv("COGENTS_LLM_PROVIDER", "openai"))
    """LLM provider to use (openrouter, openai, ollama, llamacpp, litellm)"""

    llm_model: Optional[str] = None
    """Specific model to use for LLM operations"""

    llm_config: Dict[str, Any] = Field(default_factory=dict)
    """Additional LLM configuration parameters"""

    # MCP Configuration (when mode="mcp")
    mcp_server_path: Optional[str] = None
    """Path to MCP server executable"""

    mcp_server_args: List[str] = Field(default_factory=list)
    """Arguments to pass to MCP server"""

    mcp_server_env: Dict[str, str] = Field(default_factory=dict)
    """Environment variables for MCP server"""

    # Logging Configuration
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    """Logging level for this toolkit"""

    enable_tracing: bool = Field(default_factory=lambda: os.getenv("COGENTS_ENABLE_TRACING", "false").lower() == "true")
    """Enable detailed tracing for debugging"""

    class Config:
        """Pydantic configuration"""

        arbitrary_types_allowed = True

    def get_llm_client(self, **kwargs):
        """
        Get an LLM client instance configured for this toolkit.

        Args:
            **kwargs: Additional arguments to pass to the LLM client

        Returns:
            BaseLLMClient: Configured LLM client instance
        """
        config = {**self.llm_config, **kwargs}
        if self.llm_model:
            config["chat_model"] = self.llm_model

        return get_llm_client(provider=self.llm_provider, **config)

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dict containing tool-specific configuration
        """
        return self.config.get(tool_name, {})

    def is_tool_activated(self, tool_name: str) -> bool:
        """
        Check if a specific tool is activated.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is activated, False otherwise
        """
        if self.activated_tools is None:
            return True
        return tool_name in self.activated_tools

    def update_config(self, **kwargs) -> "ToolkitConfig":
        """
        Create a new ToolkitConfig with updated parameters.

        Args:
            **kwargs: Configuration parameters to update

        Returns:
            New ToolkitConfig instance with updated parameters
        """
        data = self.model_dump()
        data.update(kwargs)
        return ToolkitConfig(**data)


def create_toolkit_config(
    name: str, mode: Literal["builtin", "mcp"] = "builtin", activated_tools: Optional[List[str]] = None, **config_params
) -> ToolkitConfig:
    """
    Convenience function to create a ToolkitConfig.

    Args:
        name: Toolkit name
        mode: Toolkit mode ('builtin' or 'mcp')
        activated_tools: List of tools to activate
        **config_params: Additional configuration parameters

    Returns:
        Configured ToolkitConfig instance
    """
    return ToolkitConfig(name=name, mode=mode, activated_tools=activated_tools, config=config_params)
