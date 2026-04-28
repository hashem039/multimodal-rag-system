# Audio Ingestion Pipeline Plan

## 1. Environment & Dependency Setup [DONE]
1.1 Add `faster-whisper` and related audio processing libraries (e.g., `pydub` or `ffmpeg`) to `requirements.txt`.
1.2 Verify `ffmpeg` installation in the local environment. (Note: ffmpeg not found locally, using mocks for verification).

## 2. Faster-Whisper Integration [DONE]
2.1 Create a utility or service class for `Faster-Whisper` initialization and transcription.
2.2 Implement a function to transcribe `.mp3` files and return text with timestamps.

## 3. Chunking & Indexing Logic [DONE]
3.1 Implement an audio-specific chunking strategy (e.g., grouping transcription segments).
3.2 Extend the `LlamaIndex` pipeline to handle audio transcripts as nodes.
3.3 Implement the logic to upsert these nodes into the existing Pinecone index.

## 4. Pipeline Integration [IN PROGRESS]
4.1 Update `src/pipeline/core.py` (or create a specific audio pipeline) to include the audio ingestion flow.
4.2 Ensure the system can differentiate between audio nodes and other future multi-modal nodes.

## 5. Verification & Testing
5.1 Write unit tests for the transcription service.
5.2 Create an integration test that processes a sample `.mp3` and verifies its presence in Pinecone.
