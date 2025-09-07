"""
Token usage tracking system for custom LLM clients.

This module provides a thread-safe token usage tracker that can be used
by custom LLM clients to report usage and by callbacks to access statistics.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from cogents_core.utils.logging import color_text, get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage data for a single LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str
    timestamp: str
    run_id: Optional[str] = None
    call_type: str = "completion"  # completion, structured, vision
    estimated: bool = False


class TokenUsageTracker:
    """Thread-safe token usage tracker."""

    def __init__(self):
        self._lock = threading.RLock()  # Use reentrant lock to avoid deadlock
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.session_start = time.time()
        self.call_count = 0
        self.usage_history: List[TokenUsage] = []

    def record_usage(self, usage: TokenUsage) -> None:
        """Record token usage from an LLM call."""
        with self._lock:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            self.call_count += 1
            self.usage_history.append(usage)

            # Log token usage in structured format for analysis with color
            model_short = usage.model_name.split("/")[-1] if "/" in usage.model_name else usage.model_name
            token_log = f"TOKENS: {usage.total_tokens} | {usage.call_type} | {model_short} | P:{usage.prompt_tokens} C:{usage.completion_tokens}{'*' if usage.estimated else ''}"
            logger.info(color_text(token_log, "magenta"))

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        with self._lock:
            return self.total_prompt_tokens + self.total_completion_tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        with self._lock:
            session_duration = time.time() - self.session_start
            return {
                "session_duration_seconds": session_duration,
                "total_calls": self.call_count,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.get_total_tokens(),
                "usage_history": [
                    {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "model_name": usage.model_name,
                        "timestamp": usage.timestamp,
                        "call_type": usage.call_type,
                        "estimated": usage.estimated,
                    }
                    for usage in self.usage_history
                ],
            }

    def reset(self) -> None:
        """Reset all usage statistics."""
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.session_start = time.time()
            self.call_count = 0
            self.usage_history.clear()
            logger.info("Token usage tracker reset")


# Global token tracker instance
_global_tracker = TokenUsageTracker()


def get_token_tracker() -> TokenUsageTracker:
    """Get the global token usage tracker."""
    return _global_tracker


def record_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str,
    call_type: str = "completion",
    run_id: Optional[str] = None,
    estimated: bool = False,
) -> None:
    """Record token usage in the global tracker."""
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        run_id=run_id,
        call_type=call_type,
        estimated=estimated,
    )
    _global_tracker.record_usage(usage)


def extract_token_usage_from_openai_response(
    response, model_name: str, call_type: str = "completion"
) -> Optional[TokenUsage]:
    """Extract token usage from OpenAI API response."""
    try:
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                model_name=model_name,
                timestamp=datetime.now().isoformat(),
                call_type=call_type,
                estimated=False,
            )
    except Exception as e:
        logger.debug(f"Could not extract token usage from response: {e}")

    return None


def estimate_token_usage(
    prompt_text: str, completion_text: str, model_name: str, call_type: str = "completion"
) -> TokenUsage:
    """Estimate token usage when actual usage is not available."""
    # Simple estimation: ~4 characters per token for English text
    prompt_tokens = max(1, len(prompt_text) // 4)
    completion_tokens = max(1, len(completion_text) // 4)

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        call_type=call_type,
        estimated=True,
    )
