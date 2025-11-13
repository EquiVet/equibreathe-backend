# app/main.py
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .processing import predict_from_video
from .schemas import PredictionResponse
import logging
logging.basicConfig(level=logging.INFO)
logging.info("🚀 Starting FastAPI app...")


app = FastAPI(title="EquiBreathe Backend", version="1.0.0")

# Allow your Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
print('BASE_DIR',BASE_DIR)
MODELS_DIR = BASE_DIR / "models"

NOSTRILS_MODEL_PATH = MODELS_DIR / "best_nostrils_model.keras"
ABDOMEN_MODEL_PATH  = MODELS_DIR / "best_abdomen_model.keras"
BOTH_MODEL_PATH     = MODELS_DIR / "best_both_model.keras"
YOLO_WEIGHTS_PATH   = MODELS_DIR / "nostrils_detector.pt"


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict/nostrils", response_model=PredictionResponse)
async def predict_nostrils(file: UploadFile = File(...)):
    if not YOLO_WEIGHTS_PATH:
        raise HTTPException(500, "YOLO weights not found/loaded.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        result = predict_from_video(
            tmp.name, "nostrils",
            nostrils_model_path=NOSTRILS_MODEL_PATH,
            abdomen_model_path=ABDOMEN_MODEL_PATH,
            yolo_weights_path=YOLO_WEIGHTS_PATH
        )
    return PredictionResponse(mode="nostrils", result=result)

@app.post("/predict/abdomen", response_model=PredictionResponse)
async def predict_abdomen(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        result = predict_from_video(
            tmp.name, "abdomen",
            nostrils_model_path=NOSTRILS_MODEL_PATH,
            abdomen_model_path=ABDOMEN_MODEL_PATH,
            yolo_weights_path=YOLO_WEIGHTS_PATH
        )
    return PredictionResponse(mode="abdomen", result=result)

@app.post("/predict/both")
async def predict_both(
    nostrils: UploadFile = File(...),
    abdomen: UploadFile  = File(...)
):
    # Return a simple JSON dict for both
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(nostrils.filename)[-1]) as t1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(abdomen.filename)[-1]) as t2:
        t1.write(await nostrils.read()); t1.flush()
        t2.write(await abdomen.read());  t2.flush()

        r1 = predict_from_video(
            t1.name, "nostrils",
            nostrils_model_path=NOSTRILS_MODEL_PATH,
            abdomen_model_path=ABDOMEN_MODEL_PATH,
            yolo_weights_path=YOLO_WEIGHTS_PATH
        )
        r2 = predict_from_video(
            t2.name, "abdomen",
            nostrils_model_path=NOSTRILS_MODEL_PATH,
            abdomen_model_path=ABDOMEN_MODEL_PATH,
            yolo_weights_path=YOLO_WEIGHTS_PATH
        )
    return {"nostrils": r1, "abdomen": r2}
