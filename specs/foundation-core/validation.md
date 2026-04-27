# Validation: Foundation & Core Architecture

## Automated Checks
- **Linting:** `ruff check .` must return no errors.
- **Dependency Install:** `pip install -r requirements.txt` must succeed.
- **Pinecone Connectivity:** Running `python src/utils/pinecone_helper.py` must successfully verify or create the index (requires `PINECONE_API_KEY`).
- **Core Initialization:** Running `python src/pipeline/core.py` must log successful initialization of LlamaIndex settings.

## Manual Checks
- Verify that `src/`, `tests/`, and `data/` exist and follow the expected hierarchy.
- Ensure `.env` is properly ignored by `.gitignore` (if initialized).
