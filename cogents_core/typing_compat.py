"""
Backward compatibility module for typing_compat.

This module has been moved to cogents_core.utils.typing_compat.
This file provides backward compatibility for existing imports.
"""

# Also expose the override function explicitly for clarity
# Import all items from the new location for backward compatibility
from cogents_core.utils.typing_compat import *  # noqa: F403, F401
from cogents_core.utils.typing_compat import override  # noqa: F401
