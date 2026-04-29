# Visual & Temporal Ingestion Pipeline Requirements

## Scope
- Implementation of a processing pipeline for Image and Video data.
- Integration of Open-Source Vision-Language Models (VLMs) via Hugging Face.
- Storage of visual embeddings and metadata in Pinecone.

## Decisions
- **VLM:** Open-source models hosted on Hugging Face (e.g., Qwen2-VL or LLaVA).
- **Video Processing:** Keyframe extraction at a fixed interval of 1 frame per second (1fps).
- **Orchestration:** LlamaIndex for indexing and retrieval management.
- **Environment:** Local Python execution.

## Context
This phase builds upon the existing foundation and audio pipeline, expanding the RAG system's capabilities to handle visual information. It enables users to "chat" with their images and videos by converting visual content into searchable text descriptions and embeddings.
