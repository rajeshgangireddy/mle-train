import json

import pytest

from src.endpoints.model_endpoint import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def single_point():
    return {
        "Temperature": 23.5,
        "Humidity": 30,
        "Light": 500,
        "CO2": 400,
        "HumidityRatio": 0.004,
    }, 0


@pytest.fixture
def multiple_data_points():
    return [
        {
            "Temperature": 23.7,
            "Humidity": 26.272,
            "Light": 585,
            "CO2": 749,
            "HumidityRatio": 0.004,
        },
        {
            "Temperature": 23.5,
            "Humidity": 30,
            "Light": 500,
            "CO2": 400,
            "HumidityRatio": 0.004,
        },
    ], [0, 0]


def test_predict_success_single_datapoint(client, single_point):
    response = client.post("/predict", json=single_point[0])
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)  # Ensure predictions are returned as a list
    assert data[0] == single_point[1]


def test_predict_success_multi_datapoints(client, multiple_data_points):
    response = client.post("/predict", json=multiple_data_points[0])
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)  # Ensure predictions are returned as a list
    assert data == multiple_data_points[1]


def test_predict_invalid_data(client):
    response = client.post("/predict", json={"invalid_feature": [1.0]})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data  # Check for error message in response
