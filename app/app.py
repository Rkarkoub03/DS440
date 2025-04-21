from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.cloud import storage
from google.cloud.storage.blob import Blob
from datetime import timedelta

app = Flask(__name__)

PROJECT_ID = "ds440-455420"
BUCKET_NAME = "garmentcode-data"
ENCODED_PREFIX = "EncodedGarmentDB"
TOP_K = 3

model = SentenceTransformer("all-MiniLM-L6-v2")
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

def generate_signed_url(gcs_uri, expiration_minutes=15):
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]
    bucket = storage_client.bucket(bucket_name)
    blob = Blob(blob_name, bucket)

    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="GET",
        response_disposition="attachment"
    )

def load_vectors_and_docs():
    print("📥 Downloading vectors and metadata from GCS...", flush=True)
    vectors_blob = bucket.blob(f"{ENCODED_PREFIX}/vectors.npy")
    docs_blob = bucket.blob(f"{ENCODED_PREFIX}/garment_docs.json")

    import io
    vectors = np.load(io.BytesIO(vectors_blob.download_as_bytes()), allow_pickle=True)
    docs = json.loads(docs_blob.download_as_string())
    return vectors, docs

def search_faiss(query, garment_docs, vectors, top_k=TOP_K):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    query_vector = model.encode([query]).astype("float32")
    _, I = index.search(query_vector, top_k)
    return [garment_docs[i] for i in I[0]]

def build_prompt(results):
    output = []
    for r in results:
        processed_images = []
        for i, url in enumerate(r["image_paths"]):
            if i == 3:  # pattern image → signed download link
                signed_url = url.replace("gs://garmentcode-data/", "https://storage.googleapis.com/garmentcode-data/")
                processed_images.append(signed_url)
            else:  # show public render/texture images
                public_url = url.replace("gs://garmentcode-data/", "https://storage.googleapis.com/garmentcode-data/")
                processed_images.append(public_url)
        output.append({
            "id": r["id"],
            "images": processed_images
        })
    return output

@app.route("/")
def index():
    return render_template("main.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    vectors, garment_docs = load_vectors_and_docs()
    results = search_faiss(user_input, garment_docs, vectors)
    structured = build_prompt(results)
    return jsonify(structured)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
