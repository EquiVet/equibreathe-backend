# app/models.py
import os
from pathlib import Path
from typing import Optional

GCS_NOSTRILS = os.getenv("GCS_NOSTRILS_MODEL")   # gs://bucket/best_nostrils_model.keras
GCS_ABDOMEN  = os.getenv("GCS_ABDOMEN_MODEL")    # gs://bucket/best_abdomen_model.keras
GCS_YOLO     = os.getenv("GCS_YOLO_WEIGHTS")     # gs://bucket/nostrils_detector.pt

LOCAL_MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

NOSTRILS_MODEL_PATH = LOCAL_MODELS_DIR / "best_nostrils_model.keras"
ABDOMEN_MODEL_PATH  = LOCAL_MODELS_DIR / "best_abdomen_model.keras"
YOLO_WEIGHTS_PATH   = LOCAL_MODELS_DIR / "nostrils_detector.pt"

def _maybe_download_from_gcs(gcs_uri: Optional[str], dest: Path):
    if not gcs_uri:
        return
    if dest.exists():
        return
    from google.cloud import storage
    assert gcs_uri.startswith("gs://"), "Expected a gs:// URI"
    _, remainder = gcs_uri.split("gs://", 1)
    bucket_name, *key_parts = remainder.split("/")
    blob_name = "/".join(key_parts)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))

def ensure_models_present():
    # If env vars are set, pull from GCS; otherwise assume baked into image
    _maybe_download_from_gcs(GCS_NOSTRILS, NOSTRILS_MODEL_PATH)
    _maybe_download_from_gcs(GCS_ABDOMEN,  ABDOMEN_MODEL_PATH)
    _maybe_download_from_gcs(GCS_YOLO,     YOLO_WEIGHTS_PATH)
    return (
        str(NOSTRILS_MODEL_PATH),
        str(ABDOMEN_MODEL_PATH),
        str(YOLO_WEIGHTS_PATH if YOLO_WEIGHTS_PATH.exists() else "")
    )
