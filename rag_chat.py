import chromadb
from chromadb.config import Settings
import ollama

# ---- CONFIG ----
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "construction_sc"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"   # change ONLY if your ollama list shows a different name


DISCLAIMER = (
    "Note: Building regulations vary by city and county. "
    "This response is general guidance and not legal advice. "
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

    context_blocks = []
    for doc, meta in zip(docs, metas):
        context_blocks.append(
            f"[Source: {meta.get('source', 'unknown')} | "
            f"Category: {meta.get('category', 'unknown')}]\n"
            f"{doc}"
        )

    return "\n\n---\n\n".join(context_blocks)

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

        # 1️⃣ Embed question
        q_embedding = embed_one(question)

        # 2️⃣ Vector search
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=5,
            include=["documents", "metadatas"]
        )

        # 3️⃣ Build context
        context = build_context(results)

        # 4️⃣ Prompt LLM
        system_prompt = (

            "You are a helpful Q&A assistant embedded on a construction and remodeling website."

            "Your primary focus is construction, remodeling, permits, and building practices in South Carolina. When relevant information exists in the provided documents, use that information first and base your response on it."

            "If a question is not related to construction, or if the documents do not contain relevant information, still answer the question normally using your general knowledge. Do not explain that the question is out of scope, and do not refuse to answer."

            "Do not imply that you are a licensed contractor, inspector, or legal authority."

            "If a question involves specific construction companies, only provide information about Builders if it appears in the provided documents."

            "If a question asks to compare Builders with another company, do not compare, evaluate, or imply differences in pricing, quality, or reputation."

            "State that comparisons are not provided, then offer general construction guidance or information about Builders if relevant."

            "Do not redirect to building codes, permits, or construction topics unless the user asks about them."

            "Keep responses natural, concise, and appropriate for a public-facing website."

            f"Always include this disclaimer at the end: {DISCLAIMER}"
        )

        user_prompt = (
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer for a homeowner or business customer (not a contractor).\n"
            "- Use plain, non-technical language.\n"
            "- Provide 3–5 short bullet points.\n"
            "- If requirements depend on the city or county, say so clearly.\n"
            "- Do NOT cite code sections or laws.\n"
            "- End with a short 'What to expect next' section.\n"
            "- Encourage contacting the company for a project-specific assessment.\n"
        )


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
