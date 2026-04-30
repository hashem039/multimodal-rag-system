# Multi-Model RAG Multimedia Data Pipeline

A unified, high-performance Retrieval-Augmented Generation (RAG) system designed to process, index, and query diverse multimedia data types, including **Audio**, **Images**, and **Video**.

---

## 🚀 Overview

This project builds a robust pipeline for multimedia understanding, allowing users to "chat" with their data regardless of its format. By combining advanced Speech-to-Text, Computer Vision, and Large Language Models, the system extracts semantic meaning from various sources and stores it in a centralized vector database for intelligent retrieval.

### Key Capabilities
- **Audio Ingestion:** High-speed transcription using `Faster-Whisper`.
- **Image Understanding:** OCR and visual context extraction via Vision-Language Models (VLMs).
- **Video Analysis:** Keyframe sampling and temporal analysis for video understanding.
- **Unified Retrieval:** Weighted multi-modal retrieval across all media types using LlamaIndex.
- **Interactive Querying:** Natural language interface for cross-modal context synthesis.

---

## 🛠️ Tech Stack

- **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/)
- **Vector Database:** [Pinecone](https://www.pinecone.io/)
- **ASR (Speech-to-Text):** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- **VLM & LLMs:** Hugging Face Inference API (Meta-Llama-3, BLIP, etc.)
- **Embeddings:** Hugging Face `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Language:** Python 3.12+
- **Linting & Formatting:** [Ruff](https://beta.ruff.rs/)

---

## 📁 Project Structure

```text
├── data/                   # Multimedia storage (audio, images, video)
├── prompts/                # Prompt templates for ingestion and validation
├── scripts/                # Standalone ingestion and query scripts
├── specs/                  # Phase-based specifications and roadmaps
├── src/
│   ├── pipeline/           # Core logic (audio, visual, query engines)
│   └── utils/              # Processors (ASR, VLM) and database helpers
├── tests/                  # Automated test suite (Pytest)
├── app.py                  # Streamlit dashboard entry point
├── main.py                 # CLI application entry point
├── pyproject.toml          # Ruff and project configuration
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.12.13
- Pinecone API Key
- Hugging Face Token (for embeddings/models)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd multi-model-rag-multimedia-data-pipeline
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # MacOS/Linux
   # .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```env
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=multimodel-rag-system
   HF_TOKEN=your_huggingface_token
   HF_LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
   HF_VLM_MODEL=Salesforce/blip-image-captioning-large
   ```

---

## 🧪 Verification & Testing

### Interactive Dashboard (Recommended)
To launch the interactive multi-modal dashboard:
```bash
streamlit run app.py
```

### CLI Querying
To run the unified query pipeline via CLI:
```bash
python scripts/query_pipeline.py "Your query here"
```

---

## 🗺️ Roadmap

- [x] **Phase 1: Foundation & Core Architecture**
- [x] **Phase 2: Audio Ingestion Pipeline**
- [x] **Phase 3: Visual & Temporal Ingestion Pipeline**
- [x] **Phase 4: Retrieval & Query Engine**
- [x] **Phase 5: Interactive Dashboard**
- [ ] **Phase 6: Optimization & Evaluation**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
