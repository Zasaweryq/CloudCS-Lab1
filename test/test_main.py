# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient


VALID_INSTANCE = {
    "airline": "United Air Lines Inc.",
    "month": 3,
    "day_of_week": 5,
    "crs_dep_time": 735,
    "crs_elapsed_time": 131.0,
    "distance": 811.0,
    "dep_delay": 22.0,
    "taxi_out": 35.0,
}


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict:
        return {"delayed": True, "probability": 0.973}

    def mock_load_model(*args, **kwargs) -> None:
        return None

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)

    from main import app
    return TestClient(app)


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_correctness(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=VALID_INSTANCE
    )
    assert response.status_code == 200
    assert "delayed" in response.json()
    assert "probability" in response.json()


def test_token_not_correctness(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer kedjkj"},
        json=VALID_INSTANCE
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_token_absent(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        json=VALID_INSTANCE
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated"
    }


def test_inference(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=VALID_INSTANCE
    )
    assert response.status_code == 200
    assert response.json() == {"delayed": True, "probability": 0.973}


@pytest.mark.parametrize("field,value", [
    ("month", 13),
    ("month", 0),
    ("day_of_week", 8),
    ("crs_dep_time", 2400),
    ("crs_elapsed_time", 0.0),
    ("distance", -100.0),
    ("taxi_out", -1.0),
])
def test_validation_out_of_range(init_test_client, field, value) -> None:
    instance = dict(VALID_INSTANCE)
    instance[field] = value

    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=instance
    )
    assert response.status_code == 422


def test_validation_missing_field(init_test_client) -> None:
    instance = dict(VALID_INSTANCE)
    del instance["airline"]

    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=instance
    )
    assert response.status_code == 422


def test_validation_wrong_type(init_test_client) -> None:
    instance = dict(VALID_INSTANCE)
    instance["month"] = "март"

    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=instance
    )
    assert response.status_code == 422


def test_negative_departure_delay_allowed(init_test_client) -> None:
    """Departure ahead of schedule is a valid input, not an error."""
    instance = dict(VALID_INSTANCE)
    instance["dep_delay"] = -12.0

    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=instance
    )
    assert response.status_code == 200
