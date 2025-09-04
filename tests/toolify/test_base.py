"""
Tests for base toolkit classes.
"""

from unittest.mock import Mock, patch

import pytest

from cogents_core.toolify.base import AsyncBaseToolkit, BaseToolkit, ToolConverter, ToolkitError
from cogents_core.toolify.config import ToolkitConfig


class MockSyncToolkit(BaseToolkit):
    """Mock synchronous toolkit for testing."""

    def get_tools_map(self):
        return {
            "tool1": lambda x: f"result1: {x}",
            "tool2": lambda x: f"result2: {x}",
            "tool3": lambda x: f"result3: {x}",
        }


class MockAsyncToolkit(AsyncBaseToolkit):
    """Mock asynchronous toolkit for testing."""

    async def get_tools_map(self):
        return {
            "async_tool1": lambda x: f"async_result1: {x}",
            "async_tool2": lambda x: f"async_result2: {x}",
        }


class TestBaseToolkit:
    """Test cases for BaseToolkit."""

    def test_initialization_with_config_dict(self):
        """Test initialization with config dictionary."""
        config_dict = {"name": "test", "llm_provider": "openai"}
        toolkit = MockSyncToolkit(config_dict)

        assert isinstance(toolkit.config, ToolkitConfig)
        assert toolkit.config.name == "test"
        assert toolkit.config.llm_provider == "openai"

    def test_initialization_with_toolkit_config(self):
        """Test initialization with ToolkitConfig object."""
        config = ToolkitConfig(name="test", llm_provider="openai")
        toolkit = MockSyncToolkit(config)

        assert toolkit.config is config
        assert toolkit.config.name == "test"

    def test_initialization_with_none_config(self):
        """Test initialization with None config."""
        toolkit = MockSyncToolkit(None)

        assert isinstance(toolkit.config, ToolkitConfig)
        assert toolkit.config.name is None

    def test_get_tools_map(self):
        """Test getting tools map."""
        toolkit = MockSyncToolkit()
        tools_map = toolkit.get_tools_map()

        assert len(tools_map) == 3
        assert "tool1" in tools_map
        assert "tool2" in tools_map
        assert "tool3" in tools_map

    def test_get_filtered_tools_map_no_filter(self):
        """Test getting filtered tools map without filter."""
        toolkit = MockSyncToolkit()
        filtered_map = toolkit.get_filtered_tools_map()

        assert len(filtered_map) == 3
        assert set(filtered_map.keys()) == {"tool1", "tool2", "tool3"}

    def test_get_filtered_tools_map_with_filter(self):
        """Test getting filtered tools map with activated tools filter."""
        config = ToolkitConfig(activated_tools=["tool1", "tool3"])
        toolkit = MockSyncToolkit(config)
        filtered_map = toolkit.get_filtered_tools_map()

        assert len(filtered_map) == 2
        assert set(filtered_map.keys()) == {"tool1", "tool3"}

    def test_get_filtered_tools_map_invalid_filter(self):
        """Test getting filtered tools map with invalid tool names."""
        config = ToolkitConfig(activated_tools=["tool1", "nonexistent"])
        toolkit = MockSyncToolkit(config)

        with pytest.raises(ToolkitError, match="Activated tools not found"):
            toolkit.get_filtered_tools_map()

    def test_call_tool_success(self):
        """Test successful tool calling."""
        toolkit = MockSyncToolkit()
        result = toolkit.call_tool("tool1", x="test")

        assert result == "result1: test"

    def test_call_tool_not_found(self):
        """Test calling non-existent tool."""
        toolkit = MockSyncToolkit()

        with pytest.raises(ToolkitError, match="Tool 'nonexistent' not found"):
            toolkit.call_tool("nonexistent", x="test")

    def test_call_tool_with_filter(self):
        """Test calling tool with activated tools filter."""
        config = ToolkitConfig(activated_tools=["tool1"])
        toolkit = MockSyncToolkit(config)

        # Should work for activated tool
        result = toolkit.call_tool("tool1", x="test")
        assert result == "result1: test"

        # Should fail for non-activated tool
        with pytest.raises(ToolkitError, match="Tool 'tool2' not found"):
            toolkit.call_tool("tool2", x="test")

    @patch("cogents_core.toolify.base.ToolConverter.function_to_langchain")
    def test_get_langchain_tools(self, mock_converter):
        """Test getting LangChain tools."""
        mock_tool = Mock()
        mock_converter.return_value = mock_tool

        toolkit = MockSyncToolkit()
        tools = toolkit.get_langchain_tools()

        assert len(tools) == 3
        assert mock_converter.call_count == 3


class TestAsyncBaseToolkit:
    """Test cases for AsyncBaseToolkit."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        toolkit = MockAsyncToolkit()

        async with toolkit as t:
            assert t is toolkit
            assert toolkit._built is True

        assert toolkit._built is False

    @pytest.mark.asyncio
    async def test_build_and_cleanup(self):
        """Test build and cleanup methods."""
        toolkit = MockAsyncToolkit()

        assert toolkit._built is False

        await toolkit.build()
        assert toolkit._built is True

        # Second build should not change state
        await toolkit.build()
        assert toolkit._built is True

        await toolkit.cleanup()
        assert toolkit._built is False

    @pytest.mark.asyncio
    async def test_get_tools_map(self):
        """Test getting async tools map."""
        toolkit = MockAsyncToolkit()
        tools_map = await toolkit.get_tools_map()

        assert len(tools_map) == 2
        assert "async_tool1" in tools_map
        assert "async_tool2" in tools_map

    @pytest.mark.asyncio
    async def test_get_filtered_tools_map(self):
        """Test getting filtered async tools map."""
        config = ToolkitConfig(activated_tools=["async_tool1"])
        toolkit = MockAsyncToolkit(config)

        filtered_map = await toolkit.get_filtered_tools_map()

        assert len(filtered_map) == 1
        assert "async_tool1" in filtered_map

    @pytest.mark.asyncio
    async def test_call_tool_async_function(self):
        """Test calling async tool function."""

        async def async_func(x):
            return f"async_result: {x}"

        class AsyncFuncToolkit(AsyncBaseToolkit):
            async def get_tools_map(self):
                return {"async_func": async_func}

        toolkit = AsyncFuncToolkit()
        result = await toolkit.call_tool("async_func", x="test")

        assert result == "async_result: test"

    @pytest.mark.asyncio
    async def test_call_tool_sync_function(self):
        """Test calling sync tool function from async toolkit."""

        def sync_func(x):
            return f"sync_result: {x}"

        class SyncFuncToolkit(AsyncBaseToolkit):
            async def get_tools_map(self):
                return {"sync_func": sync_func}

        toolkit = SyncFuncToolkit()
        result = await toolkit.call_tool("sync_func", x="test")

        assert result == "sync_result: test"


class TestToolConverter:
    """Test cases for ToolConverter."""

    def test_function_to_langchain(self):
        """Test converting function to LangChain tool."""

        def test_func(x: str) -> str:
            """Test function docstring."""
            return f"result: {x}"

        with patch("cogents_core.toolify.base.tool") as mock_tool:
            mock_tool.return_value = lambda f: f  # Mock decorator

            ToolConverter.function_to_langchain(test_func)

            mock_tool.assert_called_once_with("test_func", description="Test function docstring.")

    def test_function_to_langchain_with_custom_name_desc(self):
        """Test converting function with custom name and description."""

        def test_func(x: str) -> str:
            return f"result: {x}"

        with patch("cogents_core.toolify.base.tool") as mock_tool:
            mock_tool.return_value = lambda f: f

            ToolConverter.function_to_langchain(test_func, name="custom_name", description="custom description")

            mock_tool.assert_called_once_with("custom_name", description="custom description")
