import streamlit as st
import chromadb
from chromadb.config import Settings
import ollama

# ---- CONFIG ----
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "construction_sc"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"

DISCLAIMER = (
    "Note: Building regulations vary by city and county. "
    "This information is general guidance only and not legal advice. "
    "Always confirm requirements with your local building department."
)

# ---- HELPERS ----
def embed_one(text: str):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text
    )
    return response["embeddings"][0]

def build_context(results) -> str:
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    blocks = []
    for doc, meta in zip(docs, metas):
        blocks.append(
            f"[Source: {meta.get('source','unknown')}]\n{doc}"
        )
    return "\n\n---\n\n".join(blocks)

def answer_question(question: str) -> str:
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)

    q_embedding = embed_one(question)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    context = build_context(results)

    system_prompt = (
        "You are a customer-facing construction assistant for a South Carolina contractor website.\n"
        "Answer common homeowner and business customer questions clearly and professionally.\n"
        "Use only the provided context. If details are missing, say so.\n"
        f"Always include this disclaimer at the end: {DISCLAIMER}"
    )

    user_prompt = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "INSTRUCTIONS:\n"
        "- Use plain, non-technical language\n"
        "- Answer in 3–5 short bullet points\n"
        "- End with a short 'What to expect next' section\n"
    )

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2, "num_predict": 200}
    )

    return response["message"]["content"]

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Construction Q&A Assistant", page_icon="🧱")

st.title("🧱 Construction Q&A Assistant")
st.write("Ask a question about our construction, remodeling, or services in South Carolina.")

question = st.text_input("Your question:")

if question:
    with st.spinner("Thinking..."):
        answer = answer_question(question)
    st.markdown(answer)
