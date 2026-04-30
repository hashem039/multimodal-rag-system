import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()


def init_settings():
    """Initializes global LlamaIndex settings using Hugging Face."""
    print("Initializing LlamaIndex Settings with Hugging Face...")

    # Setting up the embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

    # Use Hugging Face Inference API for LLM
    model_name = os.getenv("HF_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    token = os.getenv("HF_TOKEN")

    if not token:
        print("Warning: HF_TOKEN not found in environment variables.")

    Settings.llm = HuggingFaceInferenceAPI(model_name=model_name, token=token)

    print(f"Settings initialized with HF Model: {model_name}")


if __name__ == "__main__":
    init_settings()
    print(f"Embedding Model: {Settings.embed_model.model_name}")
