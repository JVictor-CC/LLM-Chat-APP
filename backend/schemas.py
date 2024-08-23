from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ModelLoadSchema(BaseModel):
    model_name: str
    hf_token: Optional[str] = None


class PeftModelLoadSchema(BaseModel):
    model_name: str
    base_model: str
    hf_token: Optional[str] = None


class InferenceSchema(BaseModel):
    prompt: str
    new_tokens: Optional[int] = None


class ResponseBase(BaseModel):
    prompt: str
    response: str
    date: datetime
