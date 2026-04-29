# Implementation Plan: Retrieval & Query Engine

## Task Group 1: Core Engine Initialization
1.  Initialize LlamaIndex `PineconeVectorStore` for the existing index.
2.  Configure `VectorStoreIndex` with the appropriate embedding model (Sentence-Transformers).
3.  Set up a basic `Retriever` and `QueryEngine` to verify connectivity.

## Task Group 2: Weighted Retrieval Logic
1.  Define weighting parameters for different modalities (Audio, Image, Video).
2.  Implement a custom retriever or post-processor to apply weighted scoring to retrieved nodes.
3.  Test retrieval balance with a diverse set of multi-modal queries.

## Task Group 3: Response Synthesis & Prompt Engineering
1.  Refine the synthesis prompt to ensure the LLM correctly interprets multi-modal context.
2.  Implement grounding checks to ensure the LLM cites the source modality in its answers.
3.  Configure context window management to handle multi-modal chunks effectively.

## Task Group 4: Integration & API
1.  Expose the query engine through a clean internal API or function call.
2.  Ensure compatibility with the upcoming Streamlit dashboard.
3.  Implement basic logging for query performance and retrieval hits.
