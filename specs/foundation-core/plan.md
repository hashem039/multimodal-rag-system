# Implementation Plan: Foundation & Core Architecture

## 1. Environment & Tools Setup
- [ ] Create `pyproject.toml` with Ruff and basic project metadata.
- [ ] Create `requirements.txt` with initial dependencies (`llamaindex`, `pinecone-client`, `python-dotenv`, `ruff`, `sentence-transformers`).
- [ ] Initialize `.env.example`.

## 2. Directory Structure
- [ ] Create `src/pipeline` and `src/utils`.
- [ ] Create `tests/` directory with a sample `conftest.py`.
- [ ] Create `data/` directory with subfolders for `audio`, `images`, and `video`.

## 3. Pinecone & LlamaIndex Core
- [ ] Implement `src/utils/pinecone_helper.py` to check/create the index.
- [ ] Implement `src/pipeline/core.py` to initialize LlamaIndex `Settings`.
- [ ] Create a basic `main.py` entry point to verify the setup.

## 4. Linting & Formatting
- [ ] Run `ruff format .` and `ruff check .` to ensure the foundation is clean.
