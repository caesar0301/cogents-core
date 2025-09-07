"""
Toolkit registry for managing and discovering available toolkits.
"""

from typing import Dict, List, Optional, Type, Union

from cogents_core.utils.logging import get_logger

from .base import AsyncBaseToolkit, BaseToolkit
from .config import ToolkitConfig

logger = get_logger(__name__)


class ToolkitRegistry:
    """
    Registry for managing available toolkits.

    Provides a centralized way to register, discover, and instantiate toolkits.
    """

    _registry: Dict[str, Type[Union[BaseToolkit, AsyncBaseToolkit]]] = {}

    @classmethod
    def register(cls, name: str, toolkit_class: Type[Union[BaseToolkit, AsyncBaseToolkit]]):
        """
        Register a toolkit class.

        Args:
            name: Toolkit name for lookup
            toolkit_class: Toolkit class to register
        """
        if not issubclass(toolkit_class, (BaseToolkit, AsyncBaseToolkit)):
            raise ValueError(f"Toolkit class must inherit from BaseToolkit or AsyncBaseToolkit")

        cls._registry[name] = toolkit_class
        logger.debug(f"Registered toolkit: {name} -> {toolkit_class.__name__}")

    @classmethod
    def unregister(cls, name: str):
        """
        Unregister a toolkit.

        Args:
            name: Toolkit name to unregister
        """
        if name in cls._registry:
            del cls._registry[name]
            logger.debug(f"Unregistered toolkit: {name}")

    @classmethod
    def get_toolkit_class(cls, name: str) -> Type[Union[BaseToolkit, AsyncBaseToolkit]]:
        """
        Get a registered toolkit class by name.

        Args:
            name: Toolkit name

        Returns:
            Toolkit class

        Raises:
            KeyError: If toolkit is not registered
        """
        if name not in cls._registry:
            raise KeyError(f"Toolkit '{name}' not found. Available toolkits: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_toolkits(cls) -> List[str]:
        """
        Get list of registered toolkit names.

        Returns:
            List of toolkit names
        """
        return list(cls._registry.keys())

    @classmethod
    def create_toolkit(
        cls, name: str, config: Optional[Union[ToolkitConfig, Dict]] = None
    ) -> Union[BaseToolkit, AsyncBaseToolkit]:
        """
        Create a toolkit instance by name.

        Args:
            name: Toolkit name
            config: Toolkit configuration

        Returns:
            Toolkit instance

        Raises:
            KeyError: If toolkit is not registered
        """
        toolkit_class = cls.get_toolkit_class(name)
        return toolkit_class(config=config)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a toolkit is registered.

        Args:
            name: Toolkit name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def clear(cls):
        """Clear all registered toolkits."""
        cls._registry.clear()
        logger.debug("Cleared toolkit registry")


def register_toolkit(name: str):
    """
    Decorator for registering toolkit classes.

    Args:
        name: Toolkit name for registration

    Example:
        @register_toolkit("search")
        class SearchToolkit(AsyncBaseToolkit):
            pass
    """

    def decorator(toolkit_class: Type[Union[BaseToolkit, AsyncBaseToolkit]]):
        ToolkitRegistry.register(name, toolkit_class)
        return toolkit_class

    return decorator


def get_toolkit(name: str, config: Optional[Union[ToolkitConfig, Dict]] = None) -> Union[BaseToolkit, AsyncBaseToolkit]:
    """
    Convenience function to get a toolkit instance.

    Args:
        name: Toolkit name
        config: Toolkit configuration

    Returns:
        Toolkit instance
    """
    return ToolkitRegistry.create_toolkit(name, config)


def get_toolkits_map(
    names: Optional[List[str]] = None, configs: Optional[Dict[str, Union[ToolkitConfig, Dict]]] = None
) -> Dict[str, Union[BaseToolkit, AsyncBaseToolkit]]:
    """
    Get multiple toolkit instances as a mapping.

    Args:
        names: List of toolkit names (if None, gets all registered toolkits)
        configs: Mapping of toolkit names to their configurations

    Returns:
        Dict mapping toolkit names to instances
    """
    if names is None:
        names = ToolkitRegistry.list_toolkits()

    if configs is None:
        configs = {}

    toolkits = {}
    for name in names:
        config = configs.get(name)
        toolkits[name] = ToolkitRegistry.create_toolkit(name, config)

    return toolkits


# Auto-discovery of built-in toolkits
def _discover_builtin_toolkits():
    """
    Discover and register built-in toolkits.

    This function attempts to import and register all built-in toolkit modules.
    """
    import importlib
    import pkgutil
    from pathlib import Path

    # Get the toolkits directory
    toolkits_dir = Path(__file__).parent / "toolkits"

    if not toolkits_dir.exists():
        return

    # Import all toolkit modules
    for module_info in pkgutil.iter_modules([str(toolkits_dir)]):
        if module_info.name.startswith("_"):
            continue

        try:
            module_name = f"cogents_core.toolify.toolkits.{module_info.name}"
            importlib.import_module(module_name)
            logger.debug(f"Discovered toolkit module: {module_name}")
        except ImportError as e:
            logger.warning(f"Failed to import toolkit module {module_name}: {e}")


# Initialize built-in toolkits on import
try:
    _discover_builtin_toolkits()
except Exception as e:
    logger.warning(f"Failed to discover built-in toolkits: {e}")
