
import os
import sys
import json
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from google.cloud import storage

# === Set relative path to GCS credentials ===
key_path = os.path.join(os.path.dirname(__file__), "gcs-key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

print("🚀 Script started...", flush=True)
sys.stdout.reconfigure(line_buffering=True)

# === CONFIG ===
PROJECT_ID = "ds440-455420"
BUCKET_NAME = "garmentcode-data"
RAW_PREFIX = "GarmentCodeData_v2"
ENCODED_PREFIX = "EncodedGarmentDB"

model = SentenceTransformer("all-MiniLM-L6-v2")
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

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

def build_generic_description(attribute_list):
    sentences = []
    for item in attribute_list:
        field, value = item.split(" = ")
        label = field.split(".")[-1].replace("_", " ")
        pretty = f"{label}: {value}".capitalize()
        sentences.append(pretty)
    return "This garment includes: " + ", ".join(sentences) + "."

def load_garments():
    garment_docs = []
    garment_sets = [f"{RAW_PREFIX}/garments_5000_0/"]  # Limit to one subset for now

    for prefix in garment_sets:
        for body_type in ["default_body", "random_body"]:
            body_prefix = f"{prefix}{body_type}/"
            print(f"📁 Listing garments under: {body_prefix}", flush=True)

            blobs = bucket.list_blobs(prefix=body_prefix, page_size=100)
            garment_folders = set()
            for blob in blobs:
                parts = blob.name.split("/")
                if len(parts) >= 5 and parts[-1].endswith(".yaml") and parts[3].startswith("rand_"):
                    garment_folders.add(parts[3])

            print(f"🧵 Found garment folders: {len(garment_folders)}", flush=True)

            for folder in garment_folders:
                print(f"📂 Processing: {folder}", flush=True)

                try:
                    design_blob = bucket.blob(f"{body_prefix}{folder}/{folder}_design_params.yaml")
                    spec_blob = bucket.blob(f"{body_prefix}{folder}/{folder}_specification.json")

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

                    print(f"✅ Finished: {folder}", flush=True)

                except Exception as e:
                    print(f"❌ Skipping {folder}: {e}", flush=True)

    return garment_docs

def encode_descriptions(garment_docs):
    print("🧠 Encoding garment descriptions...", flush=True)
    descriptions = [g["description_raw"] for g in garment_docs]
    return model.encode(descriptions, convert_to_tensor=False).astype("float32")

def save_outputs(garment_docs, vectors):
    print("📤 Uploading vectors and metadata to GCS...", flush=True)

    tmp_vector = "/tmp/vectors.npy"
    tmp_docs = "/tmp/garment_docs.json"

    np.save(tmp_vector, vectors)
    with open(tmp_docs, "w") as f:
        json.dump(garment_docs, f, indent=4)

    bucket.blob(f"{ENCODED_PREFIX}/vectors.npy").upload_from_filename(tmp_vector)
    bucket.blob(f"{ENCODED_PREFIX}/garment_docs.json").upload_from_filename(tmp_docs)

    print(f"✅ Upload complete → gs://{BUCKET_NAME}/{ENCODED_PREFIX}/", flush=True)

if __name__ == "__main__":
    print("🔄 Loading garments directly from Google Cloud Storage...", flush=True)
    garment_docs = load_garments()
    print(f"📦 Total garments loaded: {len(garment_docs)}", flush=True)
    vectors = encode_descriptions(garment_docs)
    save_outputs(garment_docs, vectors)
