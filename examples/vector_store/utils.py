from cogents_core.llm import get_llm_client


def get_embed_client():
    return get_llm_client(provider="ollama", embed_model="nomic-embed-text:latest")
