# Implementation Plan: Interactive Dashboard

## Task Group 1: Streamlit Infrastructure
1. Initialize `app.py` (or `main_dashboard.py`) in the project root.
2. Set up basic Streamlit page configuration (title, layout).
3. Implement sidebar for settings or metadata display.

## Task Group 2: Unified Upload Logic
1. Create a unified file uploader supporting `.mp3`, `.wav`, `.jpg`, `.png`, and `.mp4`.
2. Implement file type detection to route files to the correct processor.
3. Add progress indicators for the ingestion process.

## Task Group 3: Pipeline Integration
1. Connect the UI to `src/pipeline/audio_pipeline.py` and `src/pipeline/visual_pipeline.py`.
2. Ensure environment variables and Pinecone indices are correctly initialized within the Streamlit context.
3. Handle exceptions and display user-friendly error messages in the UI.

## Task Group 4: Chat Interface & Query Engine
1. Implement a chat input component.
2. Integrate `src/pipeline/query_engine.py` to handle user questions.
3. Maintain session state for chat history.

## Task Group 5: UI Refinement & Validation
1. Polish the styling (Vanilla CSS if needed, otherwise Streamlit defaults).
2. Perform end-to-end testing of the upload-then-query flow.
3. Verify that the system handles multiple uploads in a single session.
