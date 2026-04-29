  Summary of Changes:
   1. Core Engine (src/pipeline/query_engine.py):
       * Implemented MultiModalQueryEngine which orchestrates retrieval and synthesis.
       * Developed WeightedMultiModalRetriever to fetch results across Audio, Image, and Video types with adjustable
         importance.
       * Corrected the MetadataFilters import path to ensure compatibility with recent LlamaIndex versions.
   2. Synthesis Logic:
       * Added a custom QA_PROMPT that instructs the LLM to identify the source modality (Audio/Image/Video) for each
         part of its answer.
   3. Flexible Core (src/pipeline/core.py):
       * Updated settings to allow switching between OpenAI and Anthropic models via environment variables
         (LLM_PROVIDER).
   4. CLI Entry Point (scripts/query_pipeline.py):
       * Created a simple script to run queries from the command line:

   1         python scripts/query_pipeline.py "What was discussed in the video?"
   5. Testing:
       * Added tests/test_query_engine.py with mock logic for the weighted retriever. (Note: These tests use mocks to
         avoid network calls, but local execution was skipped per your request regarding environment issues).