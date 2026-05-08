import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "medquad"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

def load_medquad():
    print("📥 Loading MedQuAD dataset from HuggingFace...")
    dataset = load_dataset("lavita/MedQuAD", split="train")
    print(f"✅ Loaded {len(dataset)} records")
    return dataset

def chunk_documents(dataset):
    print("✂️  Chunking documents...")
    docs = []
    for row in dataset:
        question = row.get("question", "") or ""
        answer = row.get("answer", "") or ""
        source = row.get("source", "unknown") or "unknown"
        focus = row.get("focus_area", "") or ""

        # Skip empty answers
        if not answer.strip():
            continue

        # Combine into a single context chunk
        text = f"Question: {question}\nAnswer: {answer}"

        docs.append({
            "text": text,
            "metadata": {
                "source": source,
                "focus_area": focus,
                "question": question[:200]
            }
        })

    print(f"✅ {len(docs)} valid chunks ready")
    return docs

def build_vectorstore(docs):
    print("🔢 Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("🗄️  Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  Cleared old collection")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"⚡ Embedding and storing {len(docs)} chunks in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(docs), BATCH_SIZE)):
        batch = docs[i:i + BATCH_SIZE]
        texts = [d["text"] for d in batch]
        metadatas = [d["metadata"] for d in batch]
        ids = [f"doc_{i + j}" for j in range(len(batch))]

        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    print(f"✅ Vector store built at ./{CHROMA_PATH}/")
    return collection

def test_retrieval(collection):
    print("\n🧪 Testing retrieval...")
    embedder = SentenceTransformer(EMBED_MODEL)

    test_query = "What are the symptoms of diabetes?"
    query_embedding = embedder.encode([test_query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    print(f"\nQuery: {test_query}")
    print("-" * 50)
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}:")
        print(doc[:300])
        print(f"Source: {results['metadatas'][0][i]['source']}")

if __name__ == "__main__":
    dataset = load_medquad()
    docs = chunk_documents(dataset)
    collection = build_vectorstore(docs)
    test_retrieval(collection)
    print("\n🎉 Ingestion complete! ChromaDB is ready.")