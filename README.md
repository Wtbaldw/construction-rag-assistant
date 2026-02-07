# Customer-Facing-RAG-Chatbot-for-Construction-Website
## Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install chromadb streamlit ollama
python ingest.py
streamlit run app.py


TO RUN:
python -m venv venv
venv\Scripts\activate  (if not working run .\venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python ingest.py
streamlit run app.py

ADDITIONAL INFO:
venv/ and chroma_db/ are generated locally and not committed.
Re-run ingest.py only if source documents change.