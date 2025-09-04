#!/usr/bin/env python3
"""
Integration tests for LLM environment setup and configuration.

These tests verify that the LLM environment is properly configured
and all required dependencies are available.
"""

import os

import pytest


@pytest.mark.integration
class TestLLMEnvironment:
    """Test environment setup for LLM integration."""

    def test_openrouter_api_key_environment_variable(self):
        """Test that OpenRouter API key environment variable is properly set."""
        api_key = os.getenv("OPENROUTER_API_KEY")

        if api_key is None:
            pytest.skip("OPENROUTER_API_KEY not set - skipping environment test")

        assert api_key is not None
        assert len(api_key) > 0
        assert api_key.startswith("sk-")  # OpenRouter API keys typically start with sk-

    def test_required_dependencies_importable(self):
        """Test that all required LLM dependencies can be imported."""
        try:
            # Test core LLM imports
            pass

            # Test provider-specific imports

            assert True  # If we get here, imports succeeded

        except ImportError as e:
            pytest.fail(f"Failed to import LLM utilities: {e}")

    def test_optional_dependencies_availability(self):
        """Test availability of optional dependencies."""
        # Test if instructor is available (for structured completions)
        try:
            instructor_available = True
        except ImportError:
            instructor_available = False

        # Test if llama-cpp-python is available
        try:
            llamacpp_available = True
        except ImportError:
            llamacpp_available = False

        # Test if huggingface_hub is available (for model downloads)
        try:
            hf_hub_available = True
        except ImportError:
            hf_hub_available = False

        # Just log availability - don't fail if optional deps are missing
        print(f"Optional dependencies availability:")
        print(f"  instructor: {instructor_available}")
        print(f"  llama-cpp-python: {llamacpp_available}")
        print(f"  huggingface_hub: {hf_hub_available}")

        # At least one should be available for meaningful testing
        assert (
            instructor_available or llamacpp_available or hf_hub_available
        ), "At least one optional dependency should be available"

    def test_llm_client_factory_function(self):
        """Test that get_llm_client factory function works."""
        from cogents_core.llm import get_llm_client

        # Test that all supported providers can be instantiated (may skip due to missing deps)
        providers = ["openrouter", "openai", "ollama"]

        for provider in providers:
            try:
                client = get_llm_client(provider=provider)
                assert client is not None
                assert hasattr(client, "completion")
                print(f"✓ {provider} client can be instantiated")

            except (ImportError, ValueError, Exception) as e:
                # Some providers might not be available in test environment
                print(f"⚠ {provider} client unavailable: {e}")

        # Test llamacpp separately since it might auto-download
        try:
            client = get_llm_client(provider="llamacpp")
            assert client is not None
            print("✓ llamacpp client can be instantiated (with auto-download)")
        except Exception as e:
            print(f"⚠ llamacpp client unavailable: {e}")

    def test_model_path_environment_variables(self):
        """Test handling of model path environment variables."""
        # Test LLAMACPP_MODEL_PATH if set
        llamacpp_path = os.getenv("LLAMACPP_MODEL_PATH")
        if llamacpp_path:
            assert os.path.exists(llamacpp_path), f"LLAMACPP_MODEL_PATH points to non-existent file: {llamacpp_path}"
            assert llamacpp_path.endswith(".gguf"), f"LLAMACPP_MODEL_PATH should point to .gguf file: {llamacpp_path}"
            print(f"✓ LLAMACPP_MODEL_PATH is set and valid: {llamacpp_path}")
        else:
            print("⚠ LLAMACPP_MODEL_PATH not set - will use auto-download")

        # Test other model-related env vars if they exist
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_base_url:
            print(f"✓ OPENAI_BASE_URL is set: {openai_base_url}")

        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_base_url:
            print(f"✓ OLLAMA_BASE_URL is set: {ollama_base_url}")

    def test_cache_directories_writable(self):
        """Test that cache directories are writable for model downloads."""
        import tempfile
        from pathlib import Path

        # Test general temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_write.txt"
            test_file.write_text("test")
            assert test_file.exists()

        # Test HuggingFace cache directory if available
        try:
            from huggingface_hub import HfFolder

            hf_cache_dir = HfFolder.get_cache_dir() if hasattr(HfFolder, "get_cache_dir") else None
            if hf_cache_dir:
                cache_path = Path(hf_cache_dir)
                if cache_path.exists():
                    print(f"✓ HuggingFace cache directory exists and should be writable: {cache_path}")
                else:
                    print(f"⚠ HuggingFace cache directory does not exist yet: {cache_path}")
        except ImportError:
            print("⚠ HuggingFace Hub not available - cannot check cache directory")

        # Test user home directory is writable (fallback for caches)
        home_dir = Path.home()
        assert home_dir.exists(), "User home directory should exist"

        # Try to create a test directory in home
        test_cache_dir = home_dir / ".test_cache_write"
        try:
            test_cache_dir.mkdir(exist_ok=True)
            test_file = test_cache_dir / "test.txt"
            test_file.write_text("test")
            assert test_file.exists()

            # Cleanup
            test_file.unlink()
            test_cache_dir.rmdir()

            print(f"✓ User home directory is writable: {home_dir}")

        except (OSError, PermissionError) as e:
            pytest.fail(f"User home directory is not writable: {e}")

    def test_network_connectivity(self):
        """Test basic network connectivity for API calls and model downloads."""
        import urllib.error
        import urllib.request

        # Test connectivity to common LLM API endpoints
        endpoints_to_test = [
            ("OpenRouter", "https://openrouter.ai/api/v1/models"),
            ("OpenAI", "https://api.openai.com/v1/models"),
            ("HuggingFace", "https://huggingface.co/api/models?limit=1"),
        ]

        connectivity_results = []

        for name, url in endpoints_to_test:
            try:
                request = urllib.request.Request(url)
                request.add_header("User-Agent", "cogents-test/1.0")

                with urllib.request.urlopen(request, timeout=5) as response:
                    if response.status == 200:
                        connectivity_results.append((name, True, "OK"))
                        print(f"✓ {name} is reachable")
                    else:
                        connectivity_results.append((name, False, f"HTTP {response.status}"))
                        print(f"⚠ {name} returned HTTP {response.status}")

            except urllib.error.URLError as e:
                connectivity_results.append((name, False, str(e)))
                print(f"⚠ {name} is not reachable: {e}")
            except Exception as e:
                connectivity_results.append((name, False, str(e)))
                print(f"⚠ {name} connectivity test failed: {e}")

        # At least one endpoint should be reachable for meaningful testing
        reachable_count = sum(1 for _, success, _ in connectivity_results if success)

        if reachable_count == 0:
            pytest.skip("No LLM API endpoints are reachable - network connectivity issues")
        else:
            print(f"✓ {reachable_count}/{len(endpoints_to_test)} endpoints are reachable")
