# Validation: Retrieval & Query Engine

## Success Criteria
Implementation is considered successful when the following benchmarks are met:

### 1. Retrieval Accuracy (Precision @ K)
- **Test:** Run a set of known-item queries against a multi-modal dataset.
- **Metric:** The relevant content (e.g., the specific audio segment or video keyframe) must appear in the top K (e.g., K=3) retrieved nodes for at least 80% of test cases.

### 2. Multi-Modal Synthesis
- **Test:** Ask a query that requires information from two different modalities (e.g., "What was discussed in the audio while the video showed a blue screen?").
- **Verification:** The engine must correctly synthesize a response using data from both sources.

### 3. Grounding & Attribution
- **Test:** Verify that responses include metadata or citations.
- **Verification:** Every answer should be able to trace back to its source modality and specific timestamp or image ID.

### 4. Technical Integrity
- **Linting:** `ruff` check passes for all new code.
- **Unit Tests:** All core retrieval logic is covered by automated tests in `tests/`.
