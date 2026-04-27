# Multi-Modal RAG Pipeline: Audio, Image, & Video

A simple Multi-Modal Retrieval-Augmented Generation (RAG) system that processes diverse data types—Audio (ASR), Images (OCR/VLM), and Video (VLM)—to provide grounded answers. Built with **LlamaIndex** for orchestration, **Faster-Whisper** for high-performance transcription, and **Pinecone** for scalable vector search.

---

## 🚀 Overview

This project implements a unified pipeline to ingest and query multimedia data:
- **Audio:** Transcribed using `Faster-Whisper` into searchable text chunks.
- **Images/PDFs:** Processed via Multi-modal LLMs (VLMs) for visual understanding and OCR.
- **Video:** Sampled for keyframes and analyzed using VLMs to capture temporal context.
- **UI:** A sleek, interactive **Streamlit** dashboard for file uploads and chatting with your data.

## 🛠️ Tech Stack

- **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/) (Multi-modal Indexing)
- **ASR (Speech-to-Text):** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- **Vector Database:** [Pinecone](https://www.pinecone.io/)
- **VLM (Vision-Language Models):** GPT-4o, Claude 3.5 Sonnet, or Open-Source (Qwen2-VL)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Embeddings:** sentence-transformers form HF and HF apis fo llms 
