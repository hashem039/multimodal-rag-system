# Validation: Interactive Dashboard

## Successful Implementation Criteria
- [ ] **Upload Success:** Files of all supported types (Audio, Image, Video) can be uploaded and successfully processed without crashing the app.
- [ ] **Ingestion Confirmation:** The UI provides clear feedback when a file has been successfully indexed in Pinecone.
- [ ] **Query Accuracy:** The chat interface returns relevant answers based on the content of the *most recently* uploaded file (and previously indexed data).
- [ ] **Session Stability:** The app remains responsive during long-running VLM or ASR tasks (using Streamlit spinners/progress bars).
- [ ] **Error Handling:** Invalid file types or API failures are reported gracefully to the user.

## Manual Test Cases
1. **Scenario: Audio Chat**
   - Upload `data/audio/temp.mp3`.
   - Wait for "Processing complete" message.
   - Ask a question about the audio content.
   - Verify the answer is correct.

2. **Scenario: Multi-modal Query**
   - Upload an image and a video.
   - Ask a question that requires knowledge from both or either.
   - Verify the system retrieves context correctly.
