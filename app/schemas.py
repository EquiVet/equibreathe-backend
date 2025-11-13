# app/schemas.py
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    mode: str
    result: str
