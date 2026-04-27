import os

from llama_index.core import Settings

from src.pipeline.core import init_settings
from src.utils.pinecone_helper import get_pinecone_client


def test_llama_index_settings():
    """Verify that LlamaIndex settings are correctly initialized."""
    init_settings()
    assert Settings.embed_model is not None
    assert "all-MiniLM-L6-v2" in Settings.embed_model.model_name


def test_pinecone_connection():
    """Verify that we can connect to Pinecone using the environment variable."""
    if not os.getenv("PINECONE_API_KEY"):
        import pytest

        pytest.skip("PINECONE_API_KEY not set")

    pc = get_pinecone_client()
    assert pc is not None
    # Just list indexes to verify connection
    indexes = pc.list_indexes().names()
    assert isinstance(indexes, list)
