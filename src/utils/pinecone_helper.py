import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
    return Pinecone(api_key=api_key)


def init_pinecone_index(index_name: str, dimension: int = 384, metric: str = "cosine"):
    pc = get_pinecone_client()

    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",  # Defaulting to us-east-1, should be configurable
            ),
        )
    else:
        print(f"Index {index_name} already exists.")

    return pc.Index(index_name)


if __name__ == "__main__":
    # For verification testing
    index_name = os.getenv("PINECONE_INDEX_NAME", "multimedia-rag")
    try:
        index = init_pinecone_index(index_name)
        print(f"Successfully connected to index: {index_name}")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
