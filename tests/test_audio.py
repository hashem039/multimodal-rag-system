import pytest
from unittest.mock import MagicMock, patch
from src.utils.audio_processor import AudioProcessor
from src.pipeline.audio_pipeline import AudioPipeline
from llama_index.core.schema import TextNode

@patch("src.utils.audio_processor.WhisperModel")
def test_audio_processor_transcribe(mock_whisper_model):
    # Mock segments returned by faster-whisper
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.5
    mock_segment.text = "Hello world"
    
    mock_instance = mock_whisper_model.return_value
    mock_instance.transcribe.return_value = ([mock_segment], MagicMock())

    processor = AudioProcessor()
    
    # Mocking os.path.exists to return True for the dummy file
    with patch("os.path.exists", return_value=True):
        results = processor.transcribe("dummy.mp3")

    assert len(results) == 1
    assert results[0]["text"] == "Hello world"
    assert results[0]["start"] == 0.0
    assert results[0]["end"] == 2.5

@patch("src.pipeline.audio_pipeline.AudioProcessor")
@patch("src.pipeline.audio_pipeline.get_pinecone_client")
@patch("src.pipeline.audio_pipeline.VectorStoreIndex")
@patch("src.pipeline.audio_pipeline.PineconeVectorStore")
def test_audio_pipeline_process(mock_vector_store, mock_index, mock_get_pinecone, mock_processor_class):
    # Setup mocks
    mock_processor = mock_processor_class.return_value
    mock_processor.transcribe.return_value = [
        {"start": 0.0, "end": 2.0, "text": "First segment"},
        {"start": 2.0, "end": 4.0, "text": "Second segment"}
    ]
    
    mock_pc = mock_get_pinecone.return_value
    
    pipeline = AudioPipeline(index_name="test-index")
    
    with patch("os.path.exists", return_value=True):
        nodes = pipeline.process_audio("test.mp3")

    assert len(nodes) == 2
    assert nodes[0].text == "First segment"
    assert nodes[0].metadata["start_time"] == 0.0
    assert nodes[1].metadata["file_name"] == "test.mp3"
    
    # Verify Pinecone calls
    mock_vector_store.assert_called_once()
    mock_index.assert_called_once()
