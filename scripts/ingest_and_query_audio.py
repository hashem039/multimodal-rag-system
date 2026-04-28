import os
import sys
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from src.pipeline.core import init_settings
from src.pipeline.audio_pipeline import AudioPipeline
from src.utils.pinecone_helper import get_pinecone_client

load_dotenv()

def run_ingestion_and_query(audio_path: str, query_str: str):
    # 1. Initialize LlamaIndex Settings (Embeddings, etc.)
    init_settings()
    
    # 2. Ingest the Audio File
    print(f"\n--- Phase 1: Ingesting '{audio_path}' ---")
    pipeline = AudioPipeline()
    try:
        nodes = pipeline.process_audio(audio_path)
        if not nodes:
            print("Failed to generate nodes. Check if the audio file is valid and ffmpeg is installed.")
            return
    except Exception as e:
        print(f"Ingestion error: {e}")
        return

    # 3. Query the index to verify
    print(f"\n--- Phase 2: Querying index for '{query_str}' ---")
    pc = get_pinecone_client()
    index_name = os.getenv("PINECONE_INDEX_NAME", "multimedia-rag")
    pinecone_index = pc.Index(index_name)
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    query_engine = index.as_query_engine()
    response = query_engine.query(query_str)
    
    print("\n--- Results ---")
    print(f"Query: {query_str}")
    print(f"Response: {response}")
    
    # Check source nodes to verify metadata
    print("\n--- Source Metadata Verification ---")
    for source in response.source_nodes:
        meta = source.node.metadata
        print(f"Node Text: {source.node.get_content()[:100]}...")
        print(f"File: {meta.get('file_name')} | Timestamp: {meta.get('start_time')}s - {meta.get('end_time')}s")

if __name__ == "__main__":
    # Default values for quick testing
    default_audio = "data/audio/sample.mp3"
    default_query = "What is the main topic of this audio?"
    
    path = sys.argv[1] if len(sys.argv) > 1 else default_audio
    query = sys.argv[2] if len(sys.argv) > 2 else default_query
    
    if not os.path.exists(path):
        print(f"Error: Audio file not found at {path}")
        print("Please place an .mp3 file in data/audio/ or provide a path as an argument.")
    else:
        run_ingestion_and_query(path, query)
