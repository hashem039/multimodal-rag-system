import os
# from llama_index.llms.openai import OpenAI # Placeholder for future LLM integration
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()


def init_settings():
    """Initializes global LlamaIndex settings."""
    print("Initializing LlamaIndex Settings...")

    # Setting up the embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5"
    )

    # Allow LLM configuration via environment
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "openai":
        from llama_index.llms.openai import OpenAI
        Settings.llm = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
    elif llm_provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        model = os.getenv(
            "ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"
        )
        Settings.llm = Anthropic(model=model)

    print(f"Settings initialized with {llm_provider.upper()} and embedding model.")


if __name__ == "__main__":
    init_settings()
    print(f"Embedding Model: {Settings.embed_model.model_name}")
