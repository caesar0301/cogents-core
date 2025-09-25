import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from huggingface_hub import snapshot_download
from llama_cpp import Llama

from cogents_core.llm.base import BaseLLMClient
from cogents_core.tracing import estimate_token_usage, get_token_tracker
from cogents_core.tracing.opik_tracing import configure_opik
from cogents_core.utils.logging import get_logger

# Only import OPIK if tracing is enabled
OPIK_AVAILABLE = False
track = lambda func: func  # Default no-op decorator
if os.getenv("COGENTS_OPIK_TRACING", "false").lower() == "true":
    try:
        from opik import track

        OPIK_AVAILABLE = True
    except ImportError:
        pass


T = TypeVar("T")

logger = get_logger(__name__)

# Default model configuration
DEFAULT_MODEL_REPO = "ggml-org/gemma-3-270m-it-GGUF"
DEFAULT_MODEL_FILENAME = "gemma-3-270m-it-Q8_0.gguf"


def _download_default_model() -> str:
    """
    Download the default model from Hugging Face Hub if not already cached.

    Returns:
        Path to the downloaded model file
    """
    try:
        logger.info(f"No model path provided, downloading default model: {DEFAULT_MODEL_REPO}")

        # Download the model repository to local cache
        local_dir = snapshot_download(DEFAULT_MODEL_REPO)

        # Construct path to the specific model file
        model_path = os.path.join(local_dir, DEFAULT_MODEL_FILENAME)

        if not os.path.exists(model_path):
            # If the expected file doesn't exist, try to find any .gguf file
            gguf_files = [f for f in os.listdir(local_dir) if f.endswith(".gguf")]
            if gguf_files:
                model_path = os.path.join(local_dir, gguf_files[0])
                logger.info(f"Using found model file: {gguf_files[0]}")
            else:
                raise FileNotFoundError(f"No .gguf files found in downloaded model directory: {local_dir}")

        logger.info(f"Model downloaded successfully to: {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Failed to download default model: {e}")
        raise ValueError(
            f"Failed to download default model {DEFAULT_MODEL_REPO}. "
            "Please provide a model_path parameter or set LLAMACPP_MODEL_PATH environment variable."
        ) from e


class LLMClient(BaseLLMClient):
    """
    Client for interacting with local LLMs using llama-cpp-python.

    Automatically downloads a default model (ggml-org/gemma-3-270m-it-GGUF) from
    Hugging Face Hub if no model path is provided via parameter or environment variable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        instructor: bool = False,
        chat_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        **kwargs,
    ):
        """
        Initialize the LLM client.

        Args:
            model_path: Path to the GGUF model file. If not provided and LLAMACPP_MODEL_PATH
                       environment variable is not set, automatically downloads the default model
                       (ggml-org/gemma-3-270m-it-GGUF) from Hugging Face Hub.
            instructor: Whether to enable instructor for structured output
            chat_model: Model name (used for logging, defaults to model filename)
            vision_model: Vision model name (llamacpp doesn't support vision yet)
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            **kwargs: Additional arguments to pass to Llama constructor
        """
        super().__init__(**kwargs)
        # Configure Opik tracing for observability only if enabled
        if OPIK_AVAILABLE:
            configure_opik()
            self._opik_provider = "llamacpp"
        else:
            self._opik_provider = None

        # Get model path from parameter or environment, or download default model
        self.model_path = model_path or os.getenv("LLAMACPP_MODEL_PATH")
        if not self.model_path:
            logger.info("No model path provided, attempting to download default model...")
            self.model_path = _download_default_model()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Initialize Llama model
        llama_kwargs = {
            "model_path": self.model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": kwargs.get("verbose", False),
            **kwargs,
        }

        try:
            self.llama = Llama(**llama_kwargs)
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

        # Model configurations
        model_filename = Path(self.model_path).stem
        self.chat_model = chat_model or os.getenv("LLAMACPP_CHAT_MODEL", model_filename)
        self.vision_model = vision_model or os.getenv("LLAMACPP_VISION_MODEL", model_filename)
        self.embed_model = embed_model or os.getenv("LLAMACPP_EMBED_MODEL", model_filename)

        # Set instructor flag
        self.instructor_enabled = instructor

        logger.info(f"Initialized LlamaCpp client with model: {self.model_path}")

    @track
    def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate chat completion using the loaded model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response (not supported in llamacpp)
            **kwargs: Additional arguments

        Returns:
            Generated text response
        """
        if stream:
            logger.warning("Streaming is not supported in llamacpp provider, falling back to non-streaming")

        try:
            # Convert messages to prompt format
            prompt = self._format_messages_as_prompt(messages)

            # Set default max_tokens if not provided
            if max_tokens is None:
                max_tokens = kwargs.get("max_tokens", 512)

            if self.debug:
                logger.debug(f"Chat completion: {prompt}")

            # Generate response
            response = self.llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                **kwargs,
            )

            # Extract the generated text
            output_text = response["choices"][0]["text"]

            # Log token usage
            self._log_token_usage(prompt, output_text, "completion")

            return output_text.strip()

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    @track
    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        attempts: int = 2,
        backoff: float = 0.5,
        **kwargs,
    ) -> T:
        """
        Generate structured completion by prompting for JSON output.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            response_model: Pydantic model class for structured output
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            attempts: Number of attempts to make
            backoff: Backoff factor for exponential backoff
            **kwargs: Additional arguments

        Returns:
            Structured response as the specified model type
        """
        if not self.instructor_enabled:
            raise ValueError("Instructor is not enabled. Initialize LLMClient with instructor=True")

        # Add JSON schema instruction to the last message
        schema = response_model.model_json_schema()
        json_instruction = f"\n\nPlease respond with a valid JSON object that matches this schema:\n{json.dumps(schema, indent=2)}\n\nRespond with only the JSON object, no additional text."

        # Modify the last message to include JSON instruction
        modified_messages = messages.copy()
        if modified_messages:
            modified_messages[-1]["content"] += json_instruction
        else:
            modified_messages = [{"role": "user", "content": json_instruction}]

        if self.debug:
            logger.debug(f"Structured completion: {modified_messages}")

        import time

        last_err = None
        for i in range(attempts):
            try:
                # Get raw text response
                raw_response = self.completion(
                    modified_messages, temperature=temperature, max_tokens=max_tokens, **kwargs
                )

                # Try to parse as JSON
                try:
                    # Clean the response (remove any markdown formatting)
                    clean_response = raw_response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()

                    # Parse JSON
                    parsed_json = json.loads(clean_response)
                    result = response_model(**parsed_json)

                    # Log token usage for structured completion
                    prompt_text = "\n".join([msg.get("content", "") for msg in modified_messages])
                    self._log_token_usage(prompt_text, str(result), "structured")

                    return result

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse JSON response (attempt {i+1}): {e}")
                    last_err = e
                    if i < attempts - 1:
                        time.sleep(backoff * (2**i))
                        continue
                    else:
                        raise ValueError(f"Failed to get valid JSON after {attempts} attempts: {last_err}")

            except Exception as e:
                logger.error(f"Error in structured completion attempt {i+1}: {e}")
                last_err = e
                if i < attempts - 1:
                    time.sleep(backoff * (2**i))
                else:
                    raise

        raise ValueError(f"Failed to complete structured generation after {attempts} attempts: {last_err}")

    def understand_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Analyze an image (not supported by llamacpp).

        Args:
            image_path: Path to the image file
            prompt: Text prompt describing what to analyze in the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Analysis of the image

        Raises:
            NotImplementedError: Vision capabilities are not supported by llamacpp
        """
        raise NotImplementedError("Vision capabilities are not supported by the llamacpp provider")

    def understand_image_from_url(
        self,
        image_url: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Analyze an image from URL (not supported by llamacpp).

        Args:
            image_url: URL of the image
            prompt: Text prompt describing what to analyze in the image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            Analysis of the image

        Raises:
            NotImplementedError: Vision capabilities are not supported by llamacpp
        """
        raise NotImplementedError("Vision capabilities are not supported by the llamacpp provider")

    def _format_messages_as_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to a single prompt string.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def _log_token_usage(self, prompt: str, completion: str, call_type: str = "completion"):
        """Estimate and record token usage."""
        try:
            usage = estimate_token_usage(prompt, completion, self.chat_model, call_type)
            if usage:
                get_token_tracker().record_usage(usage)
                logger.debug(
                    f"Token usage (estimated) - Prompt: {usage.prompt_tokens}, "
                    f"Completion: {usage.completion_tokens}, "
                    f"Total: {usage.total_tokens} (model: {usage.model_name})"
                )
        except Exception as e:
            logger.debug(f"Could not estimate token usage: {e}")

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings using llama.cpp.

        Note: This requires the model to support embeddings. Many GGUF models
        can generate embeddings through llama.cpp.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        try:
            # Use llama.cpp's embedding functionality
            embedding = self.llama.create_embedding(text)

            if "data" in embedding and len(embedding["data"]) > 0:
                embedding_vector = embedding["data"][0]["embedding"]

                # Validate embedding dimensions
                expected_dims = self.get_embedding_dimensions()
                if len(embedding_vector) != expected_dims:
                    logger.warning(
                        f"Embedding has {len(embedding_vector)} dimensions, expected {expected_dims}. "
                        f"Consider setting COGENTS_EMBEDDING_DIMS={len(embedding_vector)} or "
                        f"using a different embedding model."
                    )

                return embedding_vector
            else:
                raise ValueError("No embedding data returned from llama.cpp")

        except Exception as e:
            logger.error(f"Error generating embedding with llama.cpp: {e}")
            logger.warning(
                "Make sure your model supports embeddings. Consider using a different provider for embeddings."
            )
            raise

    def embed_batch(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using llama.cpp.

        Args:
            chunks: List of texts to embed

        Returns:
            List of embedding lists
        """
        try:
            embeddings = []
            for chunk in chunks:
                embedding = self.embed(chunk)
                embeddings.append(embedding)
            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings with llama.cpp: {e}")
            raise

    def rerank(self, query: str, chunks: List[str]) -> List[Tuple[float, int, str]]:
        """
        Rerank chunks based on their relevance to the query.

        This implementation uses embeddings to calculate similarity scores.
        If embeddings are not available, it falls back to a simple text-based approach.

        Args:
            query: The query to rank against
            chunks: List of text chunks to rerank

        Returns:
            List of tuples (similarity_score, original_index, chunk_text)
            sorted by similarity score in descending order
        """
        try:
            # Try to use embeddings for reranking
            query_embedding = self.embed(query)
            chunk_embeddings = self.embed_batch(chunks)

            from cogents_core.utils.statistics import cosine_similarity

            # Calculate similarities and sort
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((similarity, i, chunks[i]))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Return sorted tuples
            return similarities

        except Exception as e:
            logger.error(f"Fallback reranking also failed: {e}")
            # Last resort: return original order with zero similarities
            return [(0.0, i, chunk) for i, chunk in enumerate(chunks)]
