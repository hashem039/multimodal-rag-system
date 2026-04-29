import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import os
from src.utils.visual_processor import VisualProcessor
from src.pipeline.visual_pipeline import VisualPipeline
from llama_index.core.schema import TextNode

@pytest.fixture
def visual_processor():
    return VisualProcessor()

def test_process_image(visual_processor, tmp_path):
    # Create a dummy image
    img_path = tmp_path / "test.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    
    processed_img = visual_processor.process_image(str(img_path))
    assert isinstance(processed_img, Image.Image)
    assert processed_img.size == (100, 100)

@patch('src.pipeline.visual_pipeline.VLMHelper')
@patch('src.pipeline.visual_pipeline.get_pinecone_client')
@patch('src.pipeline.visual_pipeline.PineconeVectorStore')
@patch('src.pipeline.visual_pipeline.VectorStoreIndex')
def test_visual_pipeline_image(mock_index, mock_store, mock_pc, mock_vlm, tmp_path):
    # Setup mocks
    mock_vlm_instance = mock_vlm.return_value
    mock_vlm_instance.describe_image.return_value = "A red square."
    
    # Create dummy image
    img_path = tmp_path / "test_pipeline.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    
    pipeline = VisualPipeline(index_name="test-index")
    nodes = pipeline.process_image(str(img_path))
    
    assert len(nodes) == 1
    assert nodes[0].text == "A red square."
    assert nodes[0].metadata["type"] == "image"
    assert nodes[0].metadata["file_name"] == "test_pipeline.jpg"
    
    mock_vlm_instance.describe_image.assert_called_once()
    mock_index.assert_called_once()

@patch('src.pipeline.visual_pipeline.VisualProcessor.extract_frames')
@patch('src.pipeline.visual_pipeline.VLMHelper')
@patch('src.pipeline.visual_pipeline.get_pinecone_client')
@patch('src.pipeline.visual_pipeline.PineconeVectorStore')
@patch('src.pipeline.visual_pipeline.VectorStoreIndex')
def test_visual_pipeline_video(mock_index, mock_store, mock_pc, mock_vlm, mock_extract, tmp_path):
    # Setup mocks
    mock_vlm_instance = mock_vlm.return_value
    mock_vlm_instance.describe_image.return_value = "Frame description."
    
    mock_extract.return_value = [
        {"timestamp": 0.0, "image": Image.new('RGB', (10, 10)), "frame_index": 0},
        {"timestamp": 1.0, "image": Image.new('RGB', (10, 10)), "frame_index": 30}
    ]
    
    pipeline = VisualPipeline(index_name="test-index")
    nodes = pipeline.process_video("dummy_video.mp4", fps=1)
    
    assert len(nodes) == 2
    assert nodes[0].metadata["type"] == "video_frame"
    assert nodes[0].metadata["timestamp"] == 0.0
    assert nodes[1].metadata["timestamp"] == 1.0
    
    assert mock_vlm_instance.describe_image.call_count == 2
    mock_index.assert_called_once()
