import os

from dotenv import load_dotenv

from src.pipeline.core import init_settings
from src.utils.pinecone_helper import init_pinecone_index


def main():
    print("--- Multi-Modal RAG Pipeline Foundation ---")
    load_dotenv()

    # 1. Initialize Settings
    init_settings()

    # 2. Check Pinecone (Safe check, will fail if no API key is provided)
    index_name = os.getenv("PINECONE_INDEX_NAME")
    print(f"Verifying Pinecone Index: {index_name}...")

    if not os.getenv("PINECONE_API_KEY"):
        print("Warning: PINECONE_API_KEY not set. Skipping index verification.")
    else:
        try:
            init_pinecone_index(index_name)
            print("Pinecone verification complete.")
        except Exception as e:
            print(f"Pinecone verification failed: {e}")

    print("\nFoundation setup verified.")


if __name__ == "__main__":
    main()
