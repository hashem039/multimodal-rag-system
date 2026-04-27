# from llama_index.llms.openai import OpenAI # Placeholder for future LLM integration
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()


def init_settings():
    """Initializes global LlamaIndex settings."""
    print("Initializing LlamaIndex Settings...")

    # Setting up the embedding model
    # all-MiniLM-L6-v2 has 384 dimensions
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Placeholder for LLM - can be configured via environment
    # Settings.llm = OpenAI(model="gpt-4o")

    print("Settings initialized successfully.")


if __name__ == "__main__":
    init_settings()
    print(f"Embedding Model: {Settings.embed_model.model_name}")
