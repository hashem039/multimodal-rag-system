# Audio Ingestion Pipeline Validation

## Success Criteria
- [x] A sample `.mp3` file can be transcribed (Validated with mocks in tests/test_audio.py).
- [x] Transcription segments are correctly converted into nodes (Validated with mocks in tests/test_audio.py).
- [ ] Embeddings and metadata (including timestamps) are successfully stored in Pinecone.
- [ ] A query related to the audio content returns the correct transcribed segment.

## Validation Steps
1. **Transcription Accuracy:** Run a test with a known audio clip and compare output against a reference transcript.
2. **Integration Test:** Execute the full ingestion command for an `.mp3` file and verify no errors occur.
3. **Retrieval Verification:** Use the `LlamaIndex` query engine to retrieve information specifically from the ingested audio clip.
4. **Linting & Quality:** Ensure all new code passes `ruff` checks.
