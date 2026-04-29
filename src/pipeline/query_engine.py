import os
from typing import List, Optional

from llama_index.core import PromptTemplate, QueryBundle, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.vector_stores.pinecone import PineconeVectorStore

from src.pipeline.core import init_settings
from src.utils.pinecone_helper import get_pinecone_client

# Custom prompt to encourage modality-aware answers
QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query. \n"
    "Crucial: Identify which modality (Audio, Image, or Video) "
    "the information comes from based on the metadata (type). \n"
    "If the information comes from multiple sources, synthesize them clearly.\n"
    "Query: {query_str}\n"
    "Answer: "
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

class WeightedMultiModalRetriever(BaseRetriever):
    """
    Custom retriever that fetches results from different modalities 
    and applies weighted scoring.
    """
    def __init__(
        self, 
        index: VectorStoreIndex, 
        weights: Optional[dict] = None, 
        top_k: int = 3
    ):
        self._index = index
        self._vector_store = index.vector_store
        self._weights = weights or {"audio": 1.0, "image": 1.0, "video_frame": 1.0}
        self._top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_nodes = []
        
        for modality, weight in self._weights.items():
            # Create a filter for the specific modality
            filters = MetadataFilters(filters=[
                MetadataFilter(key="type", value=modality)
            ])
            
            # Retrieve nodes for this modality
            retriever = self._index.as_retriever(
                similarity_top_k=self._top_k,
                filters=filters
            )
            nodes = retriever.retrieve(query_bundle)
            
            # Apply weight to scores
            for node in nodes:
                node.score = (node.score or 0.0) * weight
                all_nodes.append(node)
        
        # Sort by adjusted score and return top results
        all_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return all_nodes[:self._top_k * 2] # Return a bit more for synthesis

class MultiModalQueryEngine:
    def __init__(self, index_name: str = None, weights: Optional[dict] = None):
        init_settings()
        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", "multimedia-rag"
        )
        self.pc = get_pinecone_client()
        self.pinecone_index = self.pc.Index(self.index_name)
        
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        # Setup weighted retriever
        self.retriever = WeightedMultiModalRetriever(
            index=self.index,
            weights=weights,
            top_k=3
        )
        
        # Setup response synthesizer
        self.response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=QA_PROMPT
        )
        
        # Assemble query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
        )

    def query(self, query_str: str):
        """Standard query against the entire index with weighted retrieval."""
        print(f"Weighted Multi-Modal Query: {query_str}")
        return self.query_engine.query(query_str)

if __name__ == "__main__":
    engine = MultiModalQueryEngine()
    # Test query if needed
    # response = engine.query("What is in the video?")
    # print(response)
