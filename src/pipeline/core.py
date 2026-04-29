# from llama_index.llms.openai import OpenAI # Placeholder for future LLM integration
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()


def init_settings():
    """Initializes global LlamaIndex settings."""
    print("Initializing LlamaIndex Settings...")

    # Setting up the embedding model
    # BAAI/bge-large-en-v1.5 has 1024 dimensions
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5"
    )

    # Placeholder for LLM - can be configured via environment
    # Settings.llm = OpenAI(model="gpt-4o")

    print("Settings initialized successfully.")


if __name__ == "__main__":
    init_settings()
    print(f"Embedding Model: {Settings.embed_model.model_name}")
