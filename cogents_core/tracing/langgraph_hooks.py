import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from cogents_core.tracing.token_tracker import get_token_tracker
from cogents_core.utils.logging import color_text, get_logger

logger = get_logger(__name__)


class NodeLoggingCallback(BaseCallbackHandler):
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id

    def _prefix(self) -> str:
        return f"[{self.node_id}] " if self.node_id else ""

    def on_tool_end(self, output, run_id, parent_run_id, **kwargs):
        logger.info(color_text(f"{self._prefix()}[TOOL END] output={output}", "cyan", ["dim"]))

    def on_chain_end(self, output, run_id, parent_run_id, **kwargs):
        logger.info(color_text(f"{self._prefix()}[CHAIN END] output={output}", "blue", ["dim"]))

    def on_llm_end(self, response, run_id, parent_run_id, **kwargs):
        logger.info(color_text(f"{self._prefix()}[LLM END] response={response}", "magenta", ["dim"]))

    def on_custom_event(self, event_name, payload, **kwargs):
        logger.info(color_text(f"{self._prefix()}[EVENT] {event_name}: {payload}", "yellow", ["dim"]))


class TokenUsageCallback(BaseCallbackHandler):
    """Enhanced token usage callback that works with LangGraph and custom LLM clients."""

    def __init__(self, model_name: Optional[str] = None, verbose: bool = True):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.model_name = model_name
        self.verbose = verbose
        self.session_start = time.time()
        self.llm_calls = 0
        self.token_usage_history: List[Dict] = []

        # Track custom events from our LLM clients
        self._pending_calls: Dict[str, Dict] = {}

    def on_llm_start(
        self, serialized: Dict, prompts: List[str], run_id: str, parent_run_id: Optional[str] = None, **kwargs
    ):
        """Track when LLM calls start"""
        self.llm_calls += 1

        # Store prompt info for token counting
        self._pending_calls[run_id] = {"prompts": prompts, "start_time": time.time(), "call_number": self.llm_calls}

        if self.verbose:
            logger.info(
                color_text(
                    f"[TOKEN CALLBACK] LLM call #{self.llm_calls} started (run_id: {run_id[:8]}...)", "magenta", ["dim"]
                )
            )

    def on_llm_end(self, response: LLMResult, run_id: str, parent_run_id: Optional[str] = None, **kwargs):
        """Enhanced token usage tracking with multiple extraction methods"""
        usage_data = self._extract_token_usage_from_response(response)

        # Fallback to prompt estimation if no usage data available
        if not usage_data and run_id in self._pending_calls:
            usage_data = self._estimate_token_usage(self._pending_calls[run_id], response)

        if usage_data:
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)

            # Update totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            # Store in history
            call_data = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model_name": self.model_name or "unknown",
            }
            self.token_usage_history.append(call_data)

            if self.verbose:
                self._log_token_usage(prompt_tokens, completion_tokens, total_tokens, run_id)
            else:
                # Even when not verbose, show basic token usage at info level
                token_log = f"TOKENS: {total_tokens} | langchain | unknown | P:{prompt_tokens} C:{completion_tokens}"
                logger.info(color_text(token_log, "magenta"))

        # Clean up pending call
        self._pending_calls.pop(run_id, None)

    def on_custom_event(self, name: str, data: Any, run_id: str, **kwargs):
        """Handle custom events from our LLM clients for token usage"""
        if name == "token_usage":
            self._handle_custom_token_usage(data, run_id)

    def _handle_custom_token_usage(self, data: Dict, run_id: str):
        """Handle custom token usage events from our LLM clients"""
        if isinstance(data, dict) and "usage" in data:
            usage = data["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            # Update totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            # Store in history
            call_data = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model_name": self.model_name or data.get("model", "unknown"),
                "source": "custom_event",
            }
            self.token_usage_history.append(call_data)

            if self.verbose:
                self._log_token_usage(prompt_tokens, completion_tokens, total_tokens, run_id)
            else:
                # Even when not verbose, show basic token usage at info level
                model_name = (
                    data.get("model", "unknown").split("/")[-1]
                    if "/" in data.get("model", "unknown")
                    else data.get("model", "unknown")
                )
                token_log = (
                    f"TOKENS: {total_tokens} | custom_event | {model_name} | P:{prompt_tokens} C:{completion_tokens}"
                )
                logger.info(color_text(token_log, "magenta"))

    def _extract_token_usage_from_response(self, response: LLMResult) -> Optional[Dict]:
        """Extract token usage from LangChain LLMResult"""
        usage = None

        # Method 1: LLMResult llm_output
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")

        # Method 2: LLMResult response_metadata
        if not usage and hasattr(response, "response_metadata") and response.response_metadata:
            usage = response.response_metadata.get("token_usage") or response.response_metadata.get("usage")

        # Method 3: Check generations for usage info
        if not usage and hasattr(response, "generations") and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, "generation_info") and generation.generation_info:
                        gen_usage = generation.generation_info.get("token_usage") or generation.generation_info.get(
                            "usage"
                        )
                        if gen_usage:
                            usage = gen_usage
                            break
                if usage:
                    break

        return usage

    def _estimate_token_usage(self, call_info: Dict, response: LLMResult) -> Dict:
        """Estimate token usage when actual usage is not available"""
        try:
            # Simple estimation: ~4 characters per token for English text
            prompts = call_info.get("prompts", [])
            prompt_chars = sum(len(prompt) for prompt in prompts)
            prompt_tokens = max(1, prompt_chars // 4)

            # Estimate completion tokens from response
            completion_chars = 0
            if hasattr(response, "generations"):
                for generation_list in response.generations:
                    for generation in generation_list:
                        if hasattr(generation, "text"):
                            completion_chars += len(generation.text)

            completion_tokens = max(1, completion_chars // 4)

            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated": True,
            }
        except Exception as e:
            logger.warning(f"Failed to estimate token usage: {e}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": True, "error": str(e)}

    def _log_token_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int, run_id: Optional[str] = None
    ):
        """Log token usage with detailed information"""
        run_info = f" (run_id: {run_id[:8]}...)" if run_id else ""
        model_info = f" [{self.model_name}]" if self.model_name else ""

        logger.info(color_text(f"[TOKEN USAGE]{model_info}{run_info}", "magenta", ["dim"]))
        logger.info(color_text(f"  Prompt: {prompt_tokens:,} tokens", None, ["dim"]))
        logger.info(color_text(f"  Completion: {completion_tokens:,} tokens", None, ["dim"]))
        logger.info(color_text(f"  Total: {total_tokens:,} tokens", None, ["dim"]))

        # Show session totals
        session_total = self.total_tokens()
        logger.info(color_text(f"  Session Total: {session_total:,} tokens", None, ["dim"]))

    def total_tokens(self) -> int:
        """Get total tokens used in this session"""
        return self.total_prompt_tokens + self.total_completion_tokens

    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary including custom LLM client usage"""
        session_duration = time.time() - self.session_start

        # Get data from global token tracker (custom LLM clients)
        tracker_stats = get_token_tracker().get_stats()

        # Combine callback and tracker statistics
        combined_prompt_tokens = self.total_prompt_tokens + tracker_stats.get("total_prompt_tokens", 0)
        combined_completion_tokens = self.total_completion_tokens + tracker_stats.get("total_completion_tokens", 0)
        combined_total_tokens = combined_prompt_tokens + combined_completion_tokens
        combined_calls = self.llm_calls + tracker_stats.get("total_calls", 0)

        return {
            "session_duration_seconds": session_duration,
            # Combined stats from both callback and tracker
            "total_llm_calls": combined_calls,
            "total_prompt_tokens": combined_prompt_tokens,
            "total_completion_tokens": combined_completion_tokens,
            "total_tokens": combined_total_tokens,
            # Separate stats for debugging
            "callback_stats": {
                "llm_calls": self.llm_calls,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens(),
            },
            "tracker_stats": tracker_stats,
            "model_name": self.model_name,
            "token_usage_history": self.token_usage_history,
        }

    def print_session_summary(self):
        """Print a formatted session summary"""
        summary = self.get_session_summary()

        logger.info(color_text("\n" + "=" * 50, "blue", ["dim"]))
        logger.info(color_text("TOKEN USAGE SESSION SUMMARY", "blue", ["dim"]))
        logger.info(color_text("=" * 50, "blue", ["dim"]))
        logger.info(color_text(f"Session Duration: {summary['session_duration_seconds']:.2f} seconds", None, ["dim"]))
        logger.info(color_text(f"Total LLM Calls: {summary['total_llm_calls']}", None, ["dim"]))
        logger.info(color_text(f"Total Prompt Tokens: {summary['total_prompt_tokens']:,}", None, ["dim"]))
        logger.info(color_text(f"Total Completion Tokens: {summary['total_completion_tokens']:,}", None, ["dim"]))
        logger.info(color_text(f"Total Tokens: {summary['total_tokens']:,}", None, ["dim"]))

        # Show breakdown by source
        callback_stats = summary["callback_stats"]
        tracker_stats = summary["tracker_stats"]

        if callback_stats["llm_calls"] > 0:
            logger.info(
                color_text(
                    f"  LangChain Calls: {callback_stats['llm_calls']} ({callback_stats['total_tokens']:,} tokens)",
                    None,
                    ["dim"],
                )
            )

        if tracker_stats.get("total_calls", 0) > 0:
            logger.info(
                color_text(
                    f"  Custom Client Calls: {tracker_stats['total_calls']} ({tracker_stats['total_tokens']:,} tokens)",
                    None,
                    ["dim"],
                )
            )

        if self.model_name:
            logger.info(color_text(f"Model: {self.model_name}", None, ["dim"]))

        logger.info(color_text("=" * 50, "blue", ["dim"]))

    def reset_session(self):
        """Reset all counters for a new session"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.session_start = time.time()
        self.llm_calls = 0
        self.token_usage_history = []
        self._pending_calls.clear()

        # Also reset the global token tracker
        get_token_tracker().reset()

        if self.verbose:
            logger.info(color_text("[TOKEN CALLBACK] Session reset (including custom clients)", "magenta", ["dim"]))
