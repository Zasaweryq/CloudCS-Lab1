# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.pipeline import Pipeline
from pickle import load

DELAY_PROBABILITY_THRESHOLD = 0.5


def make_inference(in_model: Pipeline, in_data: dict) -> dict:
    """Return the result of predictions for in_data using in_model."""
    frame = pd.DataFrame(in_data, index=[0]).rename(columns=str.upper)
    probability = in_model.predict_proba(frame)[0][1]

    return {
        "delayed": bool(probability >= DELAY_PROBABILITY_THRESHOLD),
        "probability": round(float(probability), 3),
    }


def load_model(path: str) -> Pipeline:
    """Return the model being read which stored on the path."""
    with open(path, "rb") as file:
        model: Pipeline = load(file)

    return model
