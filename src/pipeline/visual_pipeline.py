import os
from typing import List, Dict, Any
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from src.utils.visual_processor import VisualProcessor
from src.utils.vlm_helper import VLMHelper
from src.utils.pinecone_helper import get_pinecone_client
from llama_index.core import Settings

class VisualPipeline:
    def __init__(self, index_name: str = None):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "multimedia-rag")
        self.processor = VisualProcessor()
        self.vlm = VLMHelper()
        self.pc = get_pinecone_client()
        self.pinecone_index = self.pc.Index(self.index_name)

    def process_visual(self, file_path: str) -> List[TextNode]:
        """
        Processes an image or video file: extracts content, generates descriptions via VLM,
        creates nodes, and indexes them into Pinecone.
        """
        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return self.process_video(file_path)
        else:
            return self.process_image(file_path)

    def process_image(self, image_path: str) -> List[TextNode]:
        print(f"Processing image: {image_path}")
        image = self.processor.process_image(image_path)
        description = self.vlm.describe_image(image)
        
        if "Error from VLM API" in description or "VLM API Exception" in description:
            print(f"Warning: VLM failed for {image_path}. API Response: {description}")
            # Keep a bit more info in the placeholder for debugging
            description = f"Image from {os.path.basename(image_path)} (VLM Error: {description[:50]})"
        elif "Error:" in description:
             print(f"Warning: VLM returned generic error: {description}")
             description = f"Image from {os.path.basename(image_path)} (Error)"

        file_name = os.path.basename(image_path)
        node = TextNode(
            text=description,
            id_=f"{file_name}",
            metadata={
                "file_name": file_name,
                "file_path": image_path,
                "type": "image"
            }
        )
        
        self._index_nodes([node])
        return [node]

    def process_video(self, video_path: str, fps: int = 1) -> List[TextNode]:
        print(f"Processing video: {video_path} at {fps} fps")
        frames = self.processor.extract_frames(video_path, fps=fps)
        
        nodes = []
        file_name = os.path.basename(video_path)
        
        print(f"Extracted {len(frames)} frames. Generating descriptions...")
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)} (timestamp: {frame['timestamp']:.2f}s)...")
            description = self.vlm.describe_image(frame["image"], prompt="Describe the content of this video frame briefly.")
            
            if "Error from VLM API" in description:
                description = f"Video frame from {file_name} at {frame['timestamp']}s"

            node = TextNode(
                text=description,
                id_=f"{file_name}_frame_{frame['frame_index']}",
                metadata={
                    "file_name": file_name,
                    "file_path": video_path,
                    "timestamp": frame["timestamp"],
                    "frame_index": frame["frame_index"],
                    "type": "video_frame"
                }
            )
            nodes.append(node)
        
        if nodes:
            self._index_nodes(nodes)
        
        return nodes

    def _index_nodes(self, nodes: List[TextNode]):
        print(f"Indexing {len(nodes)} nodes to Pinecone index: {self.index_name}")
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(nodes, storage_context=storage_context)
        print("Indexing complete.")

if __name__ == "__main__":
    from src.pipeline.core import init_settings
    import sys
    
    init_settings()
    
    if len(sys.argv) > 1:
        pipeline = VisualPipeline()
        pipeline.process_visual(sys.argv[1])
    else:
        print("Usage: python src/pipeline/visual_pipeline.py <path_to_image_or_video>")
