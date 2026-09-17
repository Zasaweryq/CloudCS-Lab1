# -*- coding: utf-8 -*-
import os
from model_utils import load_model, make_inference
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field


class Instance(BaseModel):
    """Flight parameters known before arrival."""

    airline: str = Field(..., example="United Air Lines Inc.")
    month: int = Field(..., ge=1, le=12, example=3)
    day_of_week: int = Field(..., ge=1, le=7, example=5)
    crs_dep_time: int = Field(..., ge=0, le=2359, example=735)
    crs_elapsed_time: float = Field(..., gt=0, example=131.0)
    distance: float = Field(..., gt=0, example=811.0)
    dep_delay: float = Field(..., example=22.0)
    taxi_out: float = Field(..., ge=0, example=35.0)


class Prediction(BaseModel):
    """Arrival delay prediction."""

    delayed: bool
    probability: float


app = FastAPI(title="Flight Delay Prediction Service", version="1.0.0")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")
model_path: str = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")


async def is_token_correct(token: str) -> bool:
    dummy_correct_token = "00000"
    return token == dummy_correct_token


async def check_token(token: str = Depends(oauth2_scheme)) -> None:
    if not await is_token_correct(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions")
async def predictions(instance: Instance,
                      token: str = Depends(check_token)) -> Prediction:
    result = make_inference(load_model(model_path), instance.dict())

    return Prediction(**result)
