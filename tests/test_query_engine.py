import sys
from unittest.mock import MagicMock, patch

# Mock dependencies that are missing on this PC to allow test collection
mock_hf = MagicMock()
sys.modules["llama_index.embeddings.huggingface"] = mock_hf
sys.modules["sentence_transformers"] = MagicMock()

with patch("src.pipeline.core.init_settings"):
    import pytest
    from llama_index.core.schema import NodeWithScore, TextNode

    from src.pipeline.query_engine import (
        MultiModalQueryEngine,
        WeightedMultiModalRetriever,
    )

@pytest.fixture(autouse=True)
def mock_init_settings():
    with patch("src.pipeline.query_engine.init_settings") as mock:
        yield mock

@pytest.fixture
def mock_index():
    with patch("src.pipeline.query_engine.get_pinecone_client"):
        with patch("src.pipeline.query_engine.VectorStoreIndex") as mock_vsi:
            mock_index_obj = MagicMock()
            mock_vsi.from_vector_store.return_value = mock_index_obj
            yield mock_index_obj

def test_weighted_retriever_retrieve(mock_index):
    # Mock nodes for different modalities
    audio_node = NodeWithScore(
        node=TextNode(text="audio content", metadata={"type": "audio"}),
        score=0.8
    )
    image_node = NodeWithScore(
        node=TextNode(text="image content", metadata={"type": "image"}),
        score=0.7
    )
    
    # Configure index to return these nodes
    mock_retriever = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever
    
    # We expect 3 calls (one for each modality: audio, image, video_frame)
    # Return audio_node for audio, image_node for image, and empty for video_frame
    mock_retriever.retrieve.side_effect = [
        [audio_node], # audio
        [image_node], # image
        []            # video_frame
    ]
    # Patch the retrieve method because side_effect is slightly more complex
    def side_effect(query_bundle):
        pass

    retriever = WeightedMultiModalRetriever(
        index=mock_index,
        weights={"audio": 2.0, "image": 1.0, "video_frame": 1.0}
    )
    
    with patch.object(mock_index, "as_retriever") as mock_as_retriever:
        mock_as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.side_effect = [[audio_node], [image_node], []]
        
        nodes = retriever._retrieve(MagicMock())
        
        # Audio score should be 0.8 * 2.0 = 1.6
        # Image score should be 0.7 * 1.0 = 0.7
        assert len(nodes) == 2
        assert nodes[0].score == 1.6
        assert nodes[1].score == 0.7

def test_query_engine_init():
    with patch("src.pipeline.query_engine.get_pinecone_client"):
        with patch("src.pipeline.query_engine.VectorStoreIndex.from_vector_store"):
            with patch("src.pipeline.query_engine.get_response_synthesizer"):
                with patch("src.pipeline.query_engine.RetrieverQueryEngine"):
                    engine = MultiModalQueryEngine()
                    assert engine.index_name is not None
                    assert engine.query_engine is not None
