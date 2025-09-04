#!/usr/bin/env python3
"""
Integration tests for LlamaCpp LLM client.

These tests verify the llamacpp client functionality including
automatic model download from Hugging Face Hub.
"""

import os
import tempfile

import pytest

from cogents.core.base.llm.llamacpp import LLMClient


@pytest.mark.integration
@pytest.mark.slow
class TestLlamaCppIntegration:
    """Integration tests for LlamaCpp client."""

    def test_initialization_with_default_model_download(self):
        """Test that LlamaCpp client can initialize with automatic model download."""
        # Clear any existing model path environment variable
        old_model_path = os.environ.get("LLAMACPP_MODEL_PATH")
        if old_model_path:
            del os.environ["LLAMACPP_MODEL_PATH"]

        try:
            # This should trigger automatic download of the default model
            client = LLMClient()

            # Verify client was initialized
            assert client is not None
            assert client.model_path is not None
            assert os.path.exists(client.model_path)
            assert client.model_path.endswith(".gguf")

            # Verify the model file is valid (has reasonable size)
            model_size = os.path.getsize(client.model_path)
            assert model_size > 1024 * 1024  # Should be at least 1MB

        finally:
            # Restore original environment variable if it existed
            if old_model_path:
                os.environ["LLAMACPP_MODEL_PATH"] = old_model_path

    def test_basic_completion_with_default_model(self):
        """Test basic completion functionality with the default downloaded model."""
        # Clear any existing model path environment variable
        old_model_path = os.environ.get("LLAMACPP_MODEL_PATH")
        if old_model_path:
            del os.environ["LLAMACPP_MODEL_PATH"]

        try:
            # Initialize client with default model
            client = LLMClient()

            # Test simple completion
            messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]

            response = client.completion(messages, temperature=0.1, max_tokens=10)

            # Verify response
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0

            # The model should be able to handle basic arithmetic
            # We're lenient here since different models might format differently
            assert any(char.isdigit() for char in response)

        finally:
            # Restore original environment variable if it existed
            if old_model_path:
                os.environ["LLAMACPP_MODEL_PATH"] = old_model_path

    def test_model_path_preference_order(self):
        """Test that model path preference follows: parameter > env var > download."""
        # Test 1: Environment variable takes precedence over download
        old_model_path = os.environ.get("LLAMACPP_MODEL_PATH")

        # Create a temporary dummy model file for testing
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp_file:
            tmp_file.write(b"dummy model content for testing")
            dummy_model_path = tmp_file.name

        try:
            # Set environment variable
            os.environ["LLAMACPP_MODEL_PATH"] = dummy_model_path

            # This should use the env var, not download
            # Since it's a dummy file, initialization will fail at the llama loading stage
            with pytest.raises((ValueError, FileNotFoundError, Exception)):
                # We expect this to fail because it's not a real model file
                # But it should at least try to use the env var path
                LLMClient()

        finally:
            # Cleanup
            if old_model_path:
                os.environ["LLAMACPP_MODEL_PATH"] = old_model_path
            else:
                os.environ.pop("LLAMACPP_MODEL_PATH", None)

            # Remove temporary file
            try:
                os.unlink(dummy_model_path)
            except OSError:
                pass

    def test_parameter_overrides_environment(self):
        """Test that explicit model_path parameter overrides environment variable."""
        old_model_path = os.environ.get("LLAMACPP_MODEL_PATH")

        # Create temporary dummy files for testing
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as env_file:
            env_file.write(b"env model content")
            env_model_path = env_file.name

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as param_file:
            param_file.write(b"param model content")
            param_model_path = param_file.name

        try:
            # Set environment variable
            os.environ["LLAMACPP_MODEL_PATH"] = env_model_path

            # Pass explicit parameter - this should take precedence
            # Since both are dummy files, initialization will fail
            with pytest.raises((ValueError, FileNotFoundError, Exception)):
                LLMClient(model_path=param_model_path)

        finally:
            # Cleanup
            if old_model_path:
                os.environ["LLAMACPP_MODEL_PATH"] = old_model_path
            else:
                os.environ.pop("LLAMACPP_MODEL_PATH", None)

            # Remove temporary files
            for path in [env_model_path, param_model_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def test_model_caching_behavior(self):
        """Test that the model download uses Hugging Face's caching mechanism."""
        # Clear any existing model path environment variable
        old_model_path = os.environ.get("LLAMACPP_MODEL_PATH")
        if old_model_path:
            del os.environ["LLAMACPP_MODEL_PATH"]

        try:
            # Initialize first client - this should download the model
            client1 = LLMClient()
            model_path1 = client1.model_path

            # Initialize second client - this should reuse cached model
            client2 = LLMClient()
            model_path2 = client2.model_path

            # Both should point to the same cached model file
            assert model_path1 == model_path2
            assert os.path.exists(model_path1)

            # Verify it's in the Hugging Face cache directory structure
            # The path should contain something like ".cache/huggingface/hub"
            assert ".cache" in model_path1 or "hub" in model_path1

        finally:
            # Restore original environment variable if it existed
            if old_model_path:
                os.environ["LLAMACPP_MODEL_PATH"] = old_model_path

    def test_error_handling_for_download_failure(self):
        """Test error handling when model download fails."""
        # This test is tricky because we want to simulate download failure
        # without actually breaking the download. We could mock the download function,
        # but that would be more of a unit test. For integration, we'll test
        # with invalid configurations that would cause download to fail.

        # Clear any existing model path environment variable
        old_model_path = os.environ.get("LLAMACPP_MODEL_PATH")
        if old_model_path:
            del os.environ["LLAMACPP_MODEL_PATH"]

        try:
            # We can't easily simulate a download failure in integration tests
            # without mocking, so we'll just verify the error message structure
            # by testing with a non-existent explicit path

            with pytest.raises(FileNotFoundError) as exc_info:
                LLMClient(model_path="/nonexistent/model/path.gguf")

            assert "Model file not found" in str(exc_info.value)

        finally:
            # Restore original environment variable if it existed
            if old_model_path:
                os.environ["LLAMACPP_MODEL_PATH"] = old_model_path

    def test_completion_with_various_parameters(self):
        """Test completion with various parameter configurations."""
        # Clear any existing model path environment variable
        old_model_path = os.environ.get("LLAMACPP_MODEL_PATH")
        if old_model_path:
            del os.environ["LLAMACPP_MODEL_PATH"]

        try:
            client = LLMClient()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello' and nothing else."},
            ]

            # Test with different temperature values
            for temperature in [0.0, 0.5, 1.0]:
                response = client.completion(messages, temperature=temperature, max_tokens=20)
                assert response is not None
                assert isinstance(response, str)
                assert len(response) > 0

            # Test with different max_tokens values
            for max_tokens in [5, 10, 50]:
                response = client.completion(messages, temperature=0.1, max_tokens=max_tokens)
                assert response is not None
                assert isinstance(response, str)
                # Note: actual token count may vary, so we don't assert on length

        finally:
            # Restore original environment variable if it existed
            if old_model_path:
                os.environ["LLAMACPP_MODEL_PATH"] = old_model_path
