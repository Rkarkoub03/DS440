import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.cloud import storage

# === CONFIG ===
PROJECT_ID = "ds440-455420"
BUCKET_NAME = "garmentcode-data"
ENCODED_PREFIX = "EncodedGarmentDB"
TOP_K = 3

# === Setup ===
model = SentenceTransformer("all-MiniLM-L6-v2")
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

def load_vectors_and_docs():
    vectors_blob = bucket.blob(f"{ENCODED_PREFIX}/vectors.npy")
    docs_blob = bucket.blob(f"{ENCODED_PREFIX}/garment_docs.json")

    vectors = np.load(vectors_blob.download_as_bytes(), allow_pickle=True)
    docs = json.loads(docs_blob.download_as_string())
    return vectors, docs

def search_faiss(query, garment_docs, vectors, top_k=TOP_K):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    query_vector = model.encode([query]).astype("float32")
    _, I = index.search(query_vector, top_k)
    return [garment_docs[i] for i in I[0]]

def build_prompt(results):
    return "\n".join([
        f"Garment ID: {r['id']}\nImage Paths: {', '.join(r['image_paths'])}"
        for r in results
    ])

if __name__ == "__main__":
    query = input("🔍 Enter your garment query: ").strip()
    vectors, garment_docs = load_vectors_and_docs()
    results = search_faiss(query, garment_docs, vectors)
    prompt = build_prompt(results)

    print("\n--- Matching Garments ---\n")
    print(prompt)
