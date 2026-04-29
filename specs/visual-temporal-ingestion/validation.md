# Visual & Temporal Ingestion Validation

## Success Criteria
- [ ] Successfully extract frames from a sample MP4 video at 1fps.
- [ ] Generate accurate text descriptions for sample images using a Hugging Face VLM.
- [ ] Successfully index image/video frame data into Pinecone.
- [ ] Retrieve relevant video timestamps/images based on a text query (e.g., "Find the scene with a cat").

## Testing Strategy
- **Unit Tests:** Verify `visual_processor.py` correctly handles various file formats.
- **Integration Tests:** End-to-end run of the `VisualIngestionPipeline` with sample data in `data/images` and `data/video`.
- **Manual Verification:** Use a query script to ensure retrieved results match the visual content of the ingested files.
