import os
import tempfile

import streamlit as st

from src.pipeline.audio_pipeline import AudioPipeline
from src.pipeline.core import init_settings
from src.pipeline.query_engine import MultiModalQueryEngine
from src.pipeline.visual_pipeline import VisualPipeline
from src.utils.pinecone_helper import delete_all_vectors

# Page configuration
st.set_page_config(
    page_title="Multi-Modal RAG Dashboard", page_icon="🤖", layout="wide"
)

# Custom CSS for better aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .source-node {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
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

# Session state for tracking ingested files
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.info("Upload your multimedia files and chat with them!")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔥 Clear Vector Index"):
        try:
            index_name = os.getenv("PINECONE_INDEX_NAME", "multimedia-rag")
            delete_all_vectors(index_name)
            st.session_state.ingested_files = []
            st.success("Vector index cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing index: {e}")

    st.markdown("---")
    st.subheader("📚 Ingested Files")
    if not st.session_state.ingested_files:
        st.write("No files ingested yet.")
    else:
        for f in st.session_state.ingested_files:
            st.write(f"- {f}")

# Main UI
st.title("🎥 Multi-Modal RAG Pipeline")
st.markdown(
    "Chat with your **Audio**, **Images**, and **Videos** using advanced AI retrieval."
)
st.markdown("---")

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
        st.image(uploaded_file, caption="Uploaded Image", width="stretch")
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

                if uploaded_file.name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(uploaded_file.name)

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
                    st.markdown(
                        f"""
                        <div class="source-node">
                            <b>Node {i + 1} (Score: {source["score"]:.4f})</b><br>
                            Type: {source["metadata"].get("type", "N/A")}<br>
                            File: {source["metadata"].get("file_name", "N/A")}<br>
                            <hr>
                            {source["text"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


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
                    st.markdown(
                        f"""
                        <div class="source-node">
                            <b>Node {i + 1} (Score: {node.score:.4f})</b><br>
                            Type: {node.metadata.get("type", "N/A")}<br>
                            File: {node.metadata.get("file_name", "N/A")}<br>
                            <hr>
                            {node.text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

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
