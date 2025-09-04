"""
Tracing and observability modules for the cogents framework.

This package provides:
- Opik tracing configuration for LLM communications
- Token usage tracking for custom LLM clients
- LangGraph hooks and callbacks for monitoring
"""

from .langgraph_hooks import NodeLoggingCallback, TokenUsageCallback
from .opik_tracing import configure_opik, create_opik_trace, get_opik_project, is_opik_enabled
from .token_tracker import (
    TokenUsage,
    TokenUsageTracker,
    estimate_token_usage,
    extract_token_usage_from_openai_response,
    get_token_tracker,
    record_token_usage,
)

__all__ = [
    # LangGraph hooks
    "NodeLoggingCallback",
    "TokenUsageCallback",
    # Opik tracing
    "configure_opik",
    "create_opik_trace",
    "get_opik_project",
    "is_opik_enabled",
    # Token tracking
    "TokenUsage",
    "TokenUsageTracker",
    "estimate_token_usage",
    "extract_token_usage_from_openai_response",
    "get_token_tracker",
    "record_token_usage",
]
