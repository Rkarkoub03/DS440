import os
import json
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import storage

# === FINAL HARD-CODED CONFIG ===
PROJECT_ID = "ds440-455420"
BUCKET_NAME = "garmentcode-data"
DATASET_PREFIX = "GarmentCodeData_v2"
OUTPUT_DIR = "/home/rkarkoub03/Processed_data"

# === Load Embedding Model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Initialize GCS Client ===
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

# === Extract meaningful features from nested design data ===
def extract_meaningful_values(d, prefix=''):
    descriptions = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if "v" in value:
                val = value["v"]
                val_type = value.get("type", "")
                if val_type.startswith("select") and val not in [None, ""]:
                    descriptions.append(f"{full_key} = {val}")
                elif val_type == "bool":
                    descriptions.append(f"{full_key} = {'enabled' if val else 'disabled'}")
            else:
                descriptions.extend(extract_meaningful_values(value, full_key))
    return descriptions

# === Convert extracted values into a natural-language description ===
def build_generic_description(attribute_list):
    sentences = []
    for item in attribute_list:
        field, value = item.split(" = ")
        label = field.split(".")[-1].replace("_", " ")
        pretty = f"{label}: {value}".capitalize()
        sentences.append(pretty)
    return "This garment includes: " + ", ".join(sentences) + "."

# === Load and process garments from GCS ===
def load_garments():
    garment_docs = []
    garment_sets = [f"{DATASET_PREFIX}/garments_5000_{i}/" for i in range(10)]

    for prefix in garment_sets:
        for body_type in ["default_body", "random_body"]:
            body_prefix = f"{prefix}{body_type}/"
            blobs = list(bucket.list_blobs(prefix=body_prefix))

            garment_folders = set(os.path.basename(blob.name.rstrip("/")).split("/")[0] for blob in blobs if blob.name.endswith("/"))

            for folder in garment_folders:
                if not folder.startswith("rand_"):
                    continue

                try:
                    design_blob_path = f"{body_prefix}{folder}/{folder}_design_params.yaml"
                    spec_blob_path = f"{body_prefix}{folder}/{folder}_specification.json"

                    # Read directly from GCS as strings
                    design_blob = bucket.blob(design_blob_path)
                    spec_blob = bucket.blob(spec_blob_path)
                    design_data = yaml.safe_load(design_blob.download_as_string())
                    spec_data = json.loads(spec_blob.download_as_string())

                    attrs = extract_meaningful_values(design_data.get("design", {}))
                    fashion_desc = build_generic_description(attrs)

                    pattern_data = spec_data.get("pattern", {})
                    panel_dict = pattern_data.get("panels", {})
                    stitch_list = spec_data.get("stitches") or pattern_data.get("stitches") or []
                    stitch_count = len(stitch_list)

                    struct_desc = f"It consists of {len(panel_dict)} panels: {', '.join(panel_dict.keys())}. It includes {stitch_count} stitched connections between panels."
                    full_description = fashion_desc + " " + struct_desc

                    gcs_base = f"gs://{BUCKET_NAME}/{body_prefix}{folder}/"
                    garment_docs.append({
                        "id": folder,
                        "description_raw": full_description,
                        "image_paths": [
                            f"{gcs_base}{folder}_render_back.png",
                            f"{gcs_base}{folder}_render_front.png",
                            f"{gcs_base}{folder}_texture.png",
                            f"{gcs_base}{folder}_pattern.png"
                        ]
                    })

                except Exception as e:
                    print(f"Skipping {folder}: {e}")

    return garment_docs

# === Encode the text descriptions into vectors ===
def encode_descriptions(garment_docs):
    descriptions = [g["description_raw"] for g in garment_docs]
    return model.encode(descriptions, convert_to_tensor=False).astype("float32")

# === Save vector and document data locally ===
def save_outputs(garment_docs, vectors):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "vectors.npy"), "wb") as f:
        np.save(f, vectors)
    with open(os.path.join(OUTPUT_DIR, "garment_docs.json"), "w") as f:
        json.dump(garment_docs, f, indent=4)

# === Main entrypoint ===
if __name__ == "__main__":
    print("Loading garments directly from Google Cloud Storage...")
    garment_docs = load_garments()
    print(f"Loaded {len(garment_docs)} garments. Encoding descriptions...")
    vectors = encode_descriptions(garment_docs)
    save_outputs(garment_docs, vectors)
    print("✅ Saved vectors and garment docs to:", OUTPUT_DIR)
