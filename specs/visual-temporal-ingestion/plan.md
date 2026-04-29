# Visual & Temporal Ingestion Implementation Plan

## Task Group 1: Infrastructure & Utilities
1. Install necessary libraries (OpenCV for video, Pillow for images, Hugging Face Hub).
2. Create `src/utils/visual_processor.py` for handling image/video manipulation.
3. Implement `extract_frames(video_path, fps=1)` utility.

## Task Group 2: VLM Integration
1. Configure Hugging Face API access for VLM usage.
2. Implement a wrapper for image-to-text conversion using a selected open-source model (e.g., Qwen2-VL).
3. Develop logic to batch process video frames through the VLM.

## Task Group 3: Pipeline Integration
1. Create `src/pipeline/visual_pipeline.py`.
2. Implement the `VisualIngestionPipeline` class inheriting from the core pipeline logic.
3. Integrate with Pinecone for storing frame-level and image-level descriptions and embeddings.

## Task Group 4: Testing & Refinement
1. Create unit tests for image processing.
2. Create integration tests for video ingestion (sampling -> VLM -> Pinecone).
3. Verify retrieval accuracy for specific visual queries.
