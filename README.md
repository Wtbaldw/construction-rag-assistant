# Customer-Facing-RAG-Chatbot-for-Construction-Website
## Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install chromadb streamlit ollama
python ingest.py
streamlit run app.py