"""
Tests for toolkit registry system.
"""

import pytest

from cogents.core.toolify.base import AsyncBaseToolkit, BaseToolkit
from cogents.core.toolify.config import ToolkitConfig
from cogents.core.toolify.registry import ToolkitRegistry, get_toolkit, get_toolkits_map, register_toolkit


class MockSyncToolkit(BaseToolkit):
    """Mock synchronous toolkit for testing."""

    def get_tools_map(self):
        return {"mock_tool": lambda x: f"sync result: {x}"}


class MockAsyncToolkit(AsyncBaseToolkit):
    """Mock asynchronous toolkit for testing."""

    async def get_tools_map(self):
        return {"mock_tool": lambda x: f"async result: {x}"}


class TestToolkitRegistry:
    """Test cases for ToolkitRegistry."""

    @pytest.fixture(autouse=True)
    def registry_fixture(self):
        """Isolate registry state for each test without affecting global state."""
        # Create a temporary registry for testing
        original_registry = ToolkitRegistry._registry.copy()
        test_registry = {}

        # Replace the global registry with our test registry
        ToolkitRegistry._registry = test_registry

        yield

        # Restore the original registry
        ToolkitRegistry._registry = original_registry

    def test_register_toolkit(self):
        """Test toolkit registration."""
        ToolkitRegistry.register("mock_sync", MockSyncToolkit)
        ToolkitRegistry.register("mock_async", MockAsyncToolkit)

        assert ToolkitRegistry.is_registered("mock_sync")
        assert ToolkitRegistry.is_registered("mock_async")
        assert not ToolkitRegistry.is_registered("nonexistent")

    def test_register_invalid_toolkit(self):
        """Test registration of invalid toolkit class."""

        class InvalidToolkit:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseToolkit"):
            ToolkitRegistry.register("invalid", InvalidToolkit)

    def test_get_toolkit_class(self):
        """Test getting registered toolkit class."""
        ToolkitRegistry.register("mock_sync", MockSyncToolkit)

        toolkit_class = ToolkitRegistry.get_toolkit_class("mock_sync")
        assert toolkit_class is MockSyncToolkit

    def test_get_nonexistent_toolkit(self):
        """Test getting non-existent toolkit."""
        with pytest.raises(KeyError, match="Toolkit 'nonexistent' not found"):
            ToolkitRegistry.get_toolkit_class("nonexistent")

    def test_list_toolkits(self):
        """Test listing registered toolkits."""
        ToolkitRegistry.register("toolkit1", MockSyncToolkit)
        ToolkitRegistry.register("toolkit2", MockAsyncToolkit)

        toolkits = ToolkitRegistry.list_toolkits()
        assert "toolkit1" in toolkits
        assert "toolkit2" in toolkits
        assert len(toolkits) == 2

    def test_create_toolkit(self):
        """Test creating toolkit instance."""
        ToolkitRegistry.register("mock_sync", MockSyncToolkit)

        config = ToolkitConfig(name="test")
        toolkit = ToolkitRegistry.create_toolkit("mock_sync", config)

        assert isinstance(toolkit, MockSyncToolkit)
        assert toolkit.config.name == "test"

    def test_unregister_toolkit(self):
        """Test unregistering toolkit."""
        ToolkitRegistry.register("mock_sync", MockSyncToolkit)
        assert ToolkitRegistry.is_registered("mock_sync")

        ToolkitRegistry.unregister("mock_sync")
        assert not ToolkitRegistry.is_registered("mock_sync")

    def test_clear_registry(self):
        """Test clearing all registrations."""
        ToolkitRegistry.register("toolkit1", MockSyncToolkit)
        ToolkitRegistry.register("toolkit2", MockAsyncToolkit)

        assert len(ToolkitRegistry.list_toolkits()) == 2

        ToolkitRegistry.clear()
        assert len(ToolkitRegistry.list_toolkits()) == 0


class TestRegisterDecorator:
    """Test cases for register_toolkit decorator."""

    @pytest.fixture(autouse=True)
    def registry_fixture(self):
        """Isolate registry state for each test without affecting global state."""
        # Create a temporary registry for testing
        original_registry = ToolkitRegistry._registry.copy()
        test_registry = {}

        # Replace the global registry with our test registry
        ToolkitRegistry._registry = test_registry

        yield

        # Restore the original registry
        ToolkitRegistry._registry = original_registry

    def test_register_decorator(self):
        """Test toolkit registration via decorator."""

        @register_toolkit("decorated")
        class DecoratedToolkit(BaseToolkit):
            def get_tools_map(self):
                return {}

        assert ToolkitRegistry.is_registered("decorated")
        toolkit_class = ToolkitRegistry.get_toolkit_class("decorated")
        assert toolkit_class is DecoratedToolkit


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @pytest.fixture(autouse=True)
    def registry_fixture(self):
        """Isolate registry state for each test without affecting global state."""
        # Create a temporary registry for testing
        original_registry = ToolkitRegistry._registry.copy()
        test_registry = {}

        # Replace the global registry with our test registry
        ToolkitRegistry._registry = test_registry

        # Register test toolkits
        ToolkitRegistry.register("mock_sync", MockSyncToolkit)
        ToolkitRegistry.register("mock_async", MockAsyncToolkit)

        yield

        # Restore the original registry
        ToolkitRegistry._registry = original_registry

    def test_get_toolkit(self):
        """Test get_toolkit convenience function."""
        config = ToolkitConfig(name="test")
        toolkit = get_toolkit("mock_sync", config)

        assert isinstance(toolkit, MockSyncToolkit)
        assert toolkit.config.name == "test"

    def test_get_toolkits_map(self):
        """Test get_toolkits_map convenience function."""
        configs = {"mock_sync": ToolkitConfig(name="sync_config"), "mock_async": ToolkitConfig(name="async_config")}

        toolkits = get_toolkits_map(["mock_sync", "mock_async"], configs)

        assert len(toolkits) == 2
        assert isinstance(toolkits["mock_sync"], MockSyncToolkit)
        assert isinstance(toolkits["mock_async"], MockAsyncToolkit)
        assert toolkits["mock_sync"].config.name == "sync_config"
        assert toolkits["mock_async"].config.name == "async_config"

    def test_get_toolkits_map_all(self):
        """Test getting all registered toolkits."""
        toolkits = get_toolkits_map()

        assert len(toolkits) == 2
        assert "mock_sync" in toolkits
        assert "mock_async" in toolkits

    def test_get_toolkits_map_no_configs(self):
        """Test getting toolkits without specific configs."""
        toolkits = get_toolkits_map(["mock_sync"])

        assert len(toolkits) == 1
        assert isinstance(toolkits["mock_sync"], MockSyncToolkit)
        # Should use default config
        assert toolkits["mock_sync"].config.name is None
