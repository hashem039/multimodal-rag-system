import os
import tempfile

import streamlit as st

from src.pipeline.audio_pipeline import AudioPipeline
from src.pipeline.core import init_settings
from src.pipeline.query_engine import MultiModalQueryEngine
from src.pipeline.visual_pipeline import VisualPipeline

# Page configuration
st.set_page_config(
    page_title="Multi-Modal RAG Dashboard", page_icon="🤖", layout="wide"
)


# Initialize LlamaIndex settings
@st.cache_resource
def get_pipelines():
    init_settings()
    return {
        "audio": AudioPipeline(),
        "visual": VisualPipeline(),
        "query_engine": MultiModalQueryEngine(),
    }


pipelines = get_pipelines()

# Sidebar
with st.sidebar:
    st.title("Settings & Status")
    st.info("Upload your multimedia files and chat with them!")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Index (Not implemented)"):
        st.warning("Index clearing is not yet implemented.")

# Main UI
st.title("🎥 Multi-Modal RAG Pipeline")
st.markdown("---")

# Task Group 2 & 3: Upload and Processing
st.subheader("1. Upload Multimedia")
uploaded_file = st.file_uploader(
    "Drag and drop or browse files",
    type=["mp3", "wav", "jpg", "png", "mp4"],
    help="Supported: MP3, WAV, JPG, PNG, MP4",
)

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)

    # Preview
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in [".jpg", ".png"]:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    elif ext == ".mp4":
        st.video(uploaded_file)
    elif ext in [".mp3", ".wav"]:
        st.audio(uploaded_file)

    if st.button("Process File"):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save to temporary file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Route to correct pipeline
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if ext in [".mp3", ".wav"]:
                    pipelines["audio"].process_audio(tmp_path)
                elif ext in [".jpg", ".png", ".mp4"]:
                    pipelines["visual"].process_visual(tmp_path)

                st.success(f"Successfully indexed {uploaded_file.name}!")
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

st.markdown("---")

# Task Group 4: Chat Interface
st.subheader("2. Chat with your Data")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Source Context"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Node {i + 1} (Score: {source['score']:.4f})**")
                    st.write(f"Type: {source['metadata'].get('type', 'N/A')}")
                    st.write(f"File: {source['metadata'].get('file_name', 'N/A')}")
                    st.text(source["text"])
                    st.markdown("---")

# React to user input
if prompt := st.chat_input("Ask a question about your files..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = pipelines["query_engine"].query(prompt)
            st.markdown(response)

            # Display source nodes in an expander
            with st.expander("View Source Context"):
                for i, node in enumerate(response.source_nodes):
                    st.markdown(f"**Node {i + 1} (Score: {node.score:.4f})**")
                    st.write(f"Type: {node.metadata.get('type', 'N/A')}")
                    st.write(f"File: {node.metadata.get('file_name', 'N/A')}")
                    st.text(node.text)
                    st.markdown("---")

    # Add assistant response to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": str(response),
            "sources": [
                {"text": n.text, "metadata": n.metadata, "score": n.score}
                for n in response.source_nodes
            ],
        }
    )
