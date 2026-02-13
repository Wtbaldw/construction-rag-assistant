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
"""
This file tests the retrieval portion of the RAG system without generating a final AI answer.
 It connects to the persistent ChromaDB vector database, embeds a user’s question using the nomic-embed-text model via Ollama,
   and performs a semantic similarity search to retrieve the top matching document chunks.
   Instead of calling a language model to generate a response, it prints the most relevant text segments along with their source metadata (category and filename),
 allowing you to verify that the vector search and embedding process are working correctly before integrating full answer generation.
"""