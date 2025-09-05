"""
Opik tracing configuration for observability of LLM communications.

This module configures Opik tracing to monitor and analyze
all LLM interactions in the Cogents application.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def configure_opik() -> bool:
    """
    Configure Opik tracing based on environment variables.

    Environment variables:
        COGENTS_OPIK_TRACING: Global toggle for Opik tracing (default: false)
        OPIK_USE_LOCAL: Use local Opik deployment (default: true)
        OPIK_LOCAL_URL: Local Opik URL (default: http://localhost:5173)
        OPIK_API_KEY: API key for Comet ML/Opik (only needed for cloud)
        OPIK_WORKSPACE: Workspace name (optional)
        OPIK_PROJECT_NAME: Project name for organizing traces
        OPIK_URL: Custom Opik URL (for cloud deployment)
        OPIK_TRACING: Enable/disable tracing (default: true if enabled globally)

    Returns:
        bool: True if Opik was successfully configured, False otherwise
    """
    try:
        # Check global Cogents Opik tracing toggle first
        cogents_opik_enabled = os.getenv("COGENTS_OPIK_TRACING", "false").lower() == "true"

        if not cogents_opik_enabled:
            logger.debug("Opik tracing disabled via COGENTS_OPIK_TRACING=false")
            return False

        # Check if using local deployment
        use_local = os.getenv("OPIK_USE_LOCAL", "true").lower() == "true"

        # Configuration variables
        opik_api_key = os.getenv("OPIK_API_KEY")
        os.getenv("OPIK_WORKSPACE")
        opik_project = os.getenv("OPIK_PROJECT_NAME", "cogents-llm")
        opik_tracing = os.getenv("OPIK_TRACING", "true").lower() == "true"

        if not opik_tracing:
            logger.debug("Opik tracing disabled via OPIK_TRACING=false")
            return False

        # For cloud deployment, API key is required
        if not use_local and not opik_api_key:
            logger.debug("No OPIK_API_KEY found for cloud deployment, Opik tracing disabled")
            return False

        # Import opik here to avoid import errors if not installed
        import opik

        # Configure Opik
        if use_local:
            # Local deployment configuration
            local_url = os.getenv("OPIK_LOCAL_URL", "http://localhost:5173")
            opik.configure(use_local=True)
            logger.info(f"Opik tracing configured for local deployment at {local_url}, project: {opik_project}")
        else:
            # Cloud deployment configuration
            opik.configure(api_key=opik_api_key, use_local=False)
            logger.info(f"Opik tracing configured for cloud deployment, project: {opik_project}")

        return True

    except ImportError:
        logger.warning("Opik package not installed, tracing disabled")
        return False
    except Exception as e:
        logger.error(f"Failed to configure Opik: {e}")
        return False


def is_opik_enabled() -> bool:
    """
    Check if Opik tracing is currently enabled.

    Returns:
        bool: True if Opik tracing is enabled, False otherwise
    """
    try:
        # Check global Cogents Opik tracing toggle first
        cogents_opik_enabled = os.getenv("COGENTS_OPIK_TRACING", "false").lower() == "true"

        if not cogents_opik_enabled:
            return False

        import opik

        # Check if Opik is properly configured
        return opik.get_current_project_name() is not None
    except (ImportError, AttributeError):
        return False
    except Exception:
        return False


def get_opik_project() -> Optional[str]:
    """
    Get the current Opik project name.

    Returns:
        Optional[str]: Project name if configured, None otherwise
    """
    try:
        import opik

        return opik.get_current_project_name()
    except (ImportError, AttributeError):
        return None
    except Exception:
        return None


def create_opik_trace(name: str, input_data: dict = None, metadata: dict = None):
    """
    Create a new Opik trace.

    Args:
        name: Name of the trace
        input_data: Input data for the trace
        metadata: Additional metadata

    Returns:
        Opik trace object or None if tracing is disabled
    """
    if not is_opik_enabled():
        return None

    try:
        import opik

        return opik.trace(name=name, input=input_data or {}, metadata=metadata or {})
    except Exception as e:
        logger.debug(f"Failed to create Opik trace: {e}")
        return None
