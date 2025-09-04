"""
Compatibility module for typing features not available in older Python versions.
"""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import override
else:
    if sys.version_info >= (3, 12):
        from typing import override
    else:
        # Fallback for Python < 3.12
        def override(func):
            """Fallback override decorator for Python < 3.12."""
            return func
