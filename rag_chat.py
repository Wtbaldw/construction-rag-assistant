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
    "This response is general guidance and not legal advice. "
    "Always confirm requirements with your local building department."
)

# ---- HELPERS ----
def embed_one(text: str):
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def build_context(results) -> str:
    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []

    blocks = []
    for doc, meta in zip(docs, metas):
        blocks.append(
            f"[Source: {meta.get('source', 'unknown')} | "
            f"Category: {meta.get('category', 'unknown')}]\n{doc}"
        )
    return "\n\n---\n\n".join(blocks)


def is_construction_related(question: str) -> bool:
    q = question.lower()
    keywords = [
        "construction", "remodel", "renov", "addition", "build", "builder", "builders",
        "contract", "estimate", "quote", "bid", "timeline", "schedule",
        "kitchen", "bath", "bathroom", "roof", "siding", "drywall", "floor",
        "deck", "porch", "patio", "foundation", "framing", "concrete",
        "electrical", "plumbing", "hvac", "insulation",
        "window", "door", "paint", "trim",
        "permit", "inspection", "code", "zoning",
        "charleston", "south carolina", "sc",
    ]
    return any(k in q for k in keywords)


# ---- CORE API ----
def answer_question(question: str) -> str:
    construction_q = is_construction_related(question)

    context = ""
    docs_available = False

    # ✅ Only use Chroma if construction-related
    if construction_q:
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_collection(name=COLLECTION_NAME)

        q_embedding = embed_one(question)
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=5,
            include=["documents", "metadatas"]
        )

        context = build_context(results)
        docs_available = bool(context.strip())

    # ---- SYSTEM PROMPT ----
    system_prompt = f"""
You are a professional, friendly Q&A assistant embedded on a construction and remodeling website.

Primary focus: construction, remodeling, and customer questions in South Carolina, with an emphasis on helping users understand Builders and typical project expectations.

Document-first behavior:
- Use provided CONTEXT only when it is relevant.
- If relevant context exists, summarize it first and separate it from general guidance.
- If no relevant context exists, answer normally using general knowledge without explaining why.

Company rules:
- Only discuss the company "Builders" if the documents explicitly mention Builders.
- Do NOT describe, compare, recommend, or give details about any other construction companies.
- If asked to compare Builders with another company, state that company comparisons are not provided
- Never mention, name, reference, or allude to any construction company other than "Builders", even as an example.
- If asked to compare Builders to another company, clearly state that company comparisons are not provided.



Professional safety/tone:
- Do not claim to be a licensed contractor, inspector, or legal authority.
- Avoid absolute guarantees; use clear, customer-friendly language.

Output behavior:
- If the question is construction-related: respond in concise bullet points followed by a short "What to expect next" section.
- If the question is NOT construction-related: answer briefly and directly. Do NOT mention construction, permits, codes, regulations, disclaimers, or the website context.
- For construction-related questions, the "What to expect next" section must always reference Builders, such as contacting Builders, requesting a consultation, or discussing project details with the Builders team.
- Never include a "What to expect next" heading without at least one complete sentence.


Disclaimer rule:
- ONLY append this disclaimer when the question is construction-related:
{DISCLAIMER}
""".strip()

    # ---- USER PROMPT ----
    if construction_q:
        user_prompt = f"""
CONTEXT (use only if relevant):
{context if docs_available else "[No relevant document excerpts provided]"}

QUESTION:
{question}

INSTRUCTIONS:
- If context contains relevant details, start with: "From Builders / Provided Documents".
- Then include "General Guidance" if helpful.
- Use plain, non-technical language.
- Do NOT cite code sections or laws.
- End with "What to expect next".
- ONLY mention Builders if explicitly supported by the documents.
""".strip()
    else:
        user_prompt = f"""
QUESTION:
{question}

INSTRUCTIONS:
- Answer briefly and directly.
- Do not mention construction, permits, codes, or disclaimers.
""".strip()

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2}
    )

    return response["message"]["content"]

"""
This file contains the core Retrieval-Augmented Generation (RAG) logic for the chatbot.
 It determines whether a user’s question is construction-related, and if so, embeds the question using nomic-embed-text,
   retrieves the most relevant document chunks from the ChromaDB vector database, and builds a structured context.
     That context is then passed to a local LLM (llama3.2:3b via Ollama) along with a carefully designed system prompt that controls tone, safety rules, company references, and output format.
       The file enforces document-first behavior, restricts discussion to the company “Builders” when supported by the data, and conditionally appends a construction-specific disclaimer.
         This module acts as the intelligent engine that combines retrieval,
            prompt engineering, and generation into a complete customer-facing Q&A system.
"""