# Requirements: Interactive Dashboard (Phase 5)

## Scope
Build a Basic MVP interactive dashboard using Streamlit to provide a user-friendly interface for the multi-modal RAG pipeline.

## Functional Requirements
- **Unified Upload Area:** A single drag-and-drop interface for uploading Audio (mp3, wav), Image (jpg, png), and Video (mp4) files.
- **Chat Interface:** A conversational UI to query the ingested multimedia data.
- **Pipeline Integration:** Seamlessly trigger the existing audio and visual ingestion pipelines upon file upload.
- **Context-Aware Responses:** Display answers from the `QueryEngine` within the chat window.

## Technical Decisions
- **Framework:** Streamlit (as per `tech-stack.md`).
- **Interaction Model:** Files are processed immediately after upload; the index is updated, and the user can then start chatting.
- **Persistence:** Use existing Pinecone and LlamaIndex configurations.

## Context
This phase moves the project from a script-based interaction (`scripts/`) to a graphical user interface, making the RAG pipeline accessible to non-technical users.
