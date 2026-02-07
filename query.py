import chromadb
from chromadb.config import Settings
import ollama

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "construction_sc"

def embed_one(text: str):
    out = ollama.embed(
        model="nomic-embed-text",
        input=text
    )
    return out["embeddings"][0]

def main():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection(name=COLLECTION_NAME)

    print("🔎 RAG Retrieval Test (type 'exit' to quit)")

    while True:
        question = input("\nAsk a construction question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        q_emb = embed_one(question)

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=4,
            include=["documents", "metadatas"]
        )

        print("\n📌 Top matches:")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print("\n---")
            print(f"Category: {meta['category']}")
            print(f"Source: {meta['source']}")
            print(doc)

if __name__ == "__main__":
    main()
