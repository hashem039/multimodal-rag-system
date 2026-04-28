import os
from typing import List, Dict, Any
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from src.utils.audio_processor import AudioProcessor
from src.utils.pinecone_helper import get_pinecone_client
from llama_index.core import Settings

class AudioPipeline:
    def __init__(self, index_name: str = None):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "multimedia-rag")
        self.processor = AudioProcessor()
        self.pc = get_pinecone_client()
        self.pinecone_index = self.pc.Index(self.index_name)

    def process_audio(self, audio_path: str) -> List[TextNode]:
        """
        Processes an audio file: transcribes, creates nodes, and indexes them.
        """
        print(f"Starting transcription for: {audio_path}")
        segments = self.processor.transcribe(audio_path)
        
        nodes = []
        file_name = os.path.basename(audio_path)
        
        for i, segment in enumerate(segments):
            # Create a TextNode for each segment to preserve metadata like timestamps
            node = TextNode(
                text=segment["text"],
                id_=f"{file_name}_seg_{i}",
                metadata={
                    "file_name": file_name,
                    "file_path": audio_path,
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "type": "audio"
                }
            )
            nodes.append(node)
        
        if not nodes:
            print("No transcription segments found.")
            return []

        print(f"Generated {len(nodes)} nodes. Indexing to Pinecone...")
        
        # Initialize Vector Store
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Index the nodes
        VectorStoreIndex(nodes, storage_context=storage_context)
        
        print("Indexing complete.")
        return nodes

if __name__ == "__main__":
    from src.pipeline.core import init_settings
    import sys
    
    init_settings()
    
    if len(sys.argv) > 1:
        pipeline = AudioPipeline()
        pipeline.process_audio(sys.argv[1])
    else:
        print("Usage: python src/pipeline/audio_pipeline.py <path_to_audio_file>")
