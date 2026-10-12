# Construction Q&A Assistant (local RAG)

A prototype question-and-answer assistant for a South Carolina construction and remodeling website. It answers from a small set of local documents using retrieval-augmented generation (RAG), with Ollama models, a ChromaDB vector store, and a Streamlit page. Everything runs on your own machine.

## How it works

**1. Ingest (`ingest.py`)**
- Walks `ragData/` and its subfolders and reads `.txt`, `.md`, `.text` and extensionless files.
- Splits each file into 900-character chunks with 150 characters of overlap.
- Embeds the chunks with `nomic-embed-text` through Ollama, 64 per batch.
- Stores them in a persistent ChromaDB collection (`construction_sc` in `chroma_db/`). Each chunk carries metadata: source file, category (the top-level folder) and chunk index.
- Chunk IDs are a SHA-1 hash of path, index and text. Re-running deletes those IDs before adding them again, so you don't get duplicates.

**2. Answer (`rag_chat.py`)**
- A keyword check decides whether a question is about construction (remodel, permit, roof, zoning and so on).
- Construction questions: embed the question, pull the 5 closest chunks from ChromaDB, and pass them to `llama3.2:3b` as labeled context (source and category).
- Other questions skip retrieval, and the model answers briefly from general knowledge.
- A system prompt sets the rules:
  - use the documents first when they're relevant
  - mention the company ("Builders") only when the documents support it
  - never name or compare other contractors
  - never claim to be a licensed contractor or inspector
  - format construction answers as bullet points plus a "What to expect next" section, ending with a regulations disclaimer
- Temperature is 0.2.

**3. Interface (`app.py`)**
- A Streamlit page with one text box. It calls `answer_question()` and renders the reply as Markdown.

**Retrieval check (`query.py`)**
- A command-line loop that prints the top 4 matching chunks and their sources, without generating an answer. Useful for checking retrieval on its own.

**Knowledge base (`ragData/`)**
- `Services/`: services and FAQ text for the company, which is referred to only as "Builders"
- `South Carolina/`: short summaries of state permitting, licensing and code adoption, each ending in a source link
- `Universal/`: short summaries of IBC, IRC, ICC codes and OSHA, plus a regulations disclaimer

## Run it

Requires [Ollama](https://ollama.com/) running locally. The pinned `requirements.txt` was generated on Python 3.11.

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python ingest.py                # builds chroma_db/ from ragData/
streamlit run app.py            # opens the Q&A page
python query.py                 # optional: test retrieval only
```

Re-run `ingest.py` whenever the files in `ragData/` change. `chroma_db/` and `venv/` are generated locally and git-ignored.

## Project context

Built in February 2026. `Info.txt` holds the working title "Local LLM-Based Expert Systems for Construction Knowledge Assistance".

## Limitations

- **Loose keyword gate.** It uses substring matching (`"sc" in question`), so unrelated words like "describe" count as construction questions.
- **Rules live in the prompt, not the code.** The "Builders"-only rule, the answer format and the disclaimer are prompt instructions. Nothing checks that the model followed them.
- **Small knowledge base.** `ragData/` has 16 text files, and 3 of them are empty.
- **Stale chunks can pile up.** Chunk IDs include the chunk text, so editing or deleting a source file leaves its old chunks in the collection. Delete `chroma_db/` for a clean rebuild.
- **Single turn, no visible sources.** There's no conversation history, and the page shows only the answer. Source labels go to the model but aren't displayed.
- **Fixed-size chunks.** Chunks are cut by character count, not at sentence or section breaks.
- **Not wired into a site.** It runs as a standalone Streamlit page. There's no embeddable widget.
- **Heavy dependency list.** `requirements.txt` is a full environment freeze (132 packages), not a minimal list.
- **No automated tests.**

## License

There's no license file, so default copyright applies. Ask before reusing the code.
