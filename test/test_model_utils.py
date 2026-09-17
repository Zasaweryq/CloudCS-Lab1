# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps


@pytest.fixture
def create_data() -> dict:
    return {"airline": "United Air Lines Inc.", "month": 3,
            "day_of_week": 5, "crs_dep_time": 735,
            "crs_elapsed_time": 131.0, "distance": 811.0,
            "dep_delay": 22.0, "taxi_out": 35.0}


def test_make_inference_delayed(monkeypatch, create_data) -> None:
    def mock_get_probabilities(_, data: pd.DataFrame) -> list[list[float]]:
        expected = {key.upper(): value for key, value in create_data.items()}
        assert expected == {
            key: value[0] for key, value in data.to_dict("list").items()
        }
        return [[0.027, 0.973]]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict_proba", mock_get_probabilities)

    result = make_inference(in_model, create_data)
    assert result == {"delayed": True, "probability": 0.973}


def test_make_inference_on_time(monkeypatch, create_data) -> None:
    def mock_get_probabilities(*args, **kwargs) -> list[list[float]]:
        return [[0.968, 0.032]]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict_proba", mock_get_probabilities)

    result = make_inference(in_model, create_data)
    assert result == {"delayed": False, "probability": 0.032}


def test_make_inference_threshold(monkeypatch, create_data) -> None:
    """Probability exactly at the threshold is classified as delayed."""
    def mock_get_probabilities(*args, **kwargs) -> list[list[float]]:
        return [[0.5, 0.5]]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict_proba", mock_get_probabilities)

    result = make_inference(in_model, create_data)
    assert result["delayed"] is True


def test_make_inference_column_names(monkeypatch, create_data) -> None:
    """Request fields are renamed to the column names used in training."""
    captured = {}

    def mock_get_probabilities(_, data: pd.DataFrame) -> list[list[float]]:
        captured["columns"] = list(data.columns)
        return [[0.5, 0.5]]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict_proba", mock_get_probabilities)
    make_inference(in_model, create_data)

    assert captured["columns"] == ["AIRLINE", "MONTH", "DAY_OF_WEEK",
                                   "CRS_DEP_TIME", "CRS_ELAPSED_TIME",
                                   "DISTANCE", "DEP_DELAY", "TAXI_OUT"]


@pytest.fixture()
def filepath_and_data(tmpdir):
    p = tmpdir.mkdir("datadir").join("fakedmodel.pkl")
    example: str = "Test message!"
    p.write_binary(dumps(example))
    return str(p), example


def test_load_model(filepath_and_data) -> None:
    assert filepath_and_data[1] == load_model(filepath_and_data[0])
