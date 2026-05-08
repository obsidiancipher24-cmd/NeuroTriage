import os
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────
CHROMA_PATH  = "chroma_db"
COLLECTION   = "medquad"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K        = 5
GROQ_MODEL   = "llama-3.1-8b-instant"

# ── Singletons (loaded once) ─────────────────────────────
_embedder   = None
_collection = None
_groq       = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def get_collection():
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION)
    return _collection

def get_groq():
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq

# ── Core retrieval ────────────────────────────────────────
def retrieve_context(query: str, top_k: int = TOP_K) -> list[dict]:
    embedder   = get_embedder()
    collection = get_collection()

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    contexts = []
    for i, doc in enumerate(results["documents"][0]):
        contexts.append({
            "text":   doc,
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "focus":  results["metadatas"][0][i].get("focus_area", ""),
            "score":  round(1 - results["distances"][0][i], 3)
                      if results.get("distances") else None
        })
    return contexts

# ── Prompt builder ────────────────────────────────────────
def build_prompt(query: str, contexts: list[dict]) -> str:
    context_block = ""
    for i, ctx in enumerate(contexts):
        context_block += f"\n[Source {i+1}] {ctx['source']}\n{ctx['text']}\n---"

    return f"""You are NeuroTriage, a clinical AI assistant built to help route patients and provide evidence-based symptom analysis.

Use ONLY the clinical documents below to answer. Always cite which source you used. 
End every response with: "⚠️ Please consult a qualified doctor for personal medical advice."

Clinical Documents:
{context_block}

Patient Query: {query}

Evidence-Based Response:"""

# ── LLM call ─────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    groq = get_groq()
    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content

# ── Main pipeline ─────────────────────────────────────────
def run_rag_pipeline(query: str) -> dict:
    contexts = retrieve_context(query)
    prompt   = build_prompt(query, contexts)
    answer   = call_llm(prompt)

    return {
        "query":   query,
        "answer":  answer,
        "sources": [{"source": c["source"],
                     "focus":  c["focus"],
                     "score":  c["score"]} for c in contexts]
    }

# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes asthma attacks?"
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = run_rag_pipeline(q)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used:")
        for s in result["sources"]:
            print(f"  - {s['source']} (relevance score: {s['score']})")