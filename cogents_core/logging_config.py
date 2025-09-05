"""
Backward compatibility module for logging_config.

This module has been moved to cogents_core.utils.logging_config.
This file provides backward compatibility for existing imports.
"""

# Also expose the main functions explicitly for clarity
# Re-export the logger to maintain compatibility with any direct access
# Import all items from the new location for backward compatibility
from cogents_core.utils.logging_config import *  # noqa: F403, F401
from cogents_core.utils.logging_config import logger  # noqa: F401
from cogents_core.utils.logging_config import (  # noqa: F401
    color_text,
    debug_color,
    error_color,
    get_logger,
    info_color,
    setup_logging,
    warning_color,
)
