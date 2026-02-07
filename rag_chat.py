import chromadb
from chromadb.config import Settings
import ollama

# ---- CONFIG ----
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "construction_sc"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"  # change ONLY if your ollama list shows a different name

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

    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(
            f"[Source: {meta.get('source', 'unknown')} | "
            f"Category: {meta.get('category', 'unknown')}]\n"
            f"{doc}"
        )
    return "\n\n---\n\n".join(context_blocks)


def has_relevant_docs(results) -> bool:
    docs = results.get("documents", [[]])[0] if results else []
    return bool(docs)


def is_construction_related(question: str) -> bool:
    q = question.lower()
    keywords = [
        "construction", "remodel", "renov", "addition", "build", "builder", "builders",
        "contract", "estimate", "quote", "bid", "timeline", "schedule",
        "kitchen", "bath", "bathroom", "roof", "siding", "drywall", "floor", "flooring",
        "deck", "porch", "patio", "foundation", "framing", "concrete",
        "electrical", "plumbing", "hvac", "insulation",
        "window", "door", "paint", "trim",
        "permit", "inspection", "code", "zoning",
        "charleston", "south carolina", "sc",
    ]
    return any(k in q for k in keywords)


# ---- MAIN CHAT LOOP ----
def main():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=COLLECTION_NAME)

    print("🧱 Construction RAG Chat (type 'exit' to quit)")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        construction_q = is_construction_related(question)

        context = ""
        docs_available = False

        # ✅ ONLY query Chroma for construction-related questions
        if construction_q:
            q_embedding = embed_one(question)
            results = collection.query(
                query_embeddings=[q_embedding],
                n_results=5,
                include=["documents", "metadatas"]
            )
            context = build_context(results)
            docs_available = has_relevant_docs(results)

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
- If asked to compare Builders with another company, state that company comparisons are not provided, then give general advice on comparing bids.

Professional safety/tone:
- Do not claim to be a licensed contractor, inspector, or legal authority.
- Avoid absolute guarantees; use clear, customer-friendly language.

Output behavior:
- If the question is construction-related: respond in concise bullet points followed by a short "What to expect next" section.
- If the question is NOT construction-related: answer briefly and directly. Do NOT mention construction, permits, codes, regulations, disclaimers, or the website context.

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

        print("\nAssistant:\n")
        print(response["message"]["content"])


if __name__ == "__main__":
    main()
