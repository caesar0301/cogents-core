"""
Tests for toolkit configuration system.
"""

from cogents_core.toolify.config import ToolkitConfig, create_toolkit_config


class TestToolkitConfig:
    """Test cases for ToolkitConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ToolkitConfig()

        assert config.mode == "builtin"
        assert config.name is None
        assert config.activated_tools is None
        assert config.config == {}
        assert config.llm_provider == "openai"
        assert config.llm_model is None
        assert config.llm_config == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ToolkitConfig(
            mode="mcp",
            name="test_toolkit",
            activated_tools=["tool1", "tool2"],
            llm_provider="openai",
            llm_model="gpt-4",
            config={"api_key": "test_key"},
        )

        assert config.mode == "mcp"
        assert config.name == "test_toolkit"
        assert config.activated_tools == ["tool1", "tool2"]
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.config["api_key"] == "test_key"

    def test_get_tool_config(self):
        """Test getting tool-specific configuration."""
        config = ToolkitConfig(config={"tool1": {"param1": "value1"}, "tool2": {"param2": "value2"}})

        assert config.get_tool_config("tool1") == {"param1": "value1"}
        assert config.get_tool_config("tool2") == {"param2": "value2"}
        assert config.get_tool_config("nonexistent") == {}

    def test_is_tool_activated(self):
        """Test tool activation checking."""
        # No filter - all tools activated
        config = ToolkitConfig()
        assert config.is_tool_activated("any_tool") is True

        # With filter - only specified tools activated
        config = ToolkitConfig(activated_tools=["tool1", "tool2"])
        assert config.is_tool_activated("tool1") is True
        assert config.is_tool_activated("tool2") is True
        assert config.is_tool_activated("tool3") is False

    def test_update_config(self):
        """Test configuration updating."""
        config = ToolkitConfig(name="original", llm_provider="openai")

        updated = config.update_config(name="updated", llm_model="gpt-4")

        # Original should be unchanged
        assert config.name == "original"
        assert config.llm_model is None

        # Updated should have new values
        assert updated.name == "updated"
        assert updated.llm_provider == "openai"  # Preserved
        assert updated.llm_model == "gpt-4"  # New

    def test_mcp_configuration(self):
        """Test MCP-specific configuration."""
        config = ToolkitConfig(
            mode="mcp",
            mcp_server_path="/path/to/server",
            mcp_server_args=["--arg1", "--arg2"],
            mcp_server_env={"ENV_VAR": "value"},
        )

        assert config.mode == "mcp"
        assert config.mcp_server_path == "/path/to/server"
        assert config.mcp_server_args == ["--arg1", "--arg2"]
        assert config.mcp_server_env == {"ENV_VAR": "value"}


class TestCreateToolkitConfig:
    """Test cases for create_toolkit_config function."""

    def test_basic_creation(self):
        """Test basic toolkit config creation."""
        config = create_toolkit_config(
            name="test_toolkit", mode="builtin", activated_tools=["tool1", "tool2"], api_key="test_key"
        )

        assert config.name == "test_toolkit"
        assert config.mode == "builtin"
        assert config.activated_tools == ["tool1", "tool2"]
        assert config.config["api_key"] == "test_key"

    def test_mcp_creation(self):
        """Test MCP toolkit config creation."""
        config = create_toolkit_config(name="mcp_toolkit", mode="mcp", server_path="/path/to/server")

        assert config.name == "mcp_toolkit"
        assert config.mode == "mcp"
        assert config.config["server_path"] == "/path/to/server"
