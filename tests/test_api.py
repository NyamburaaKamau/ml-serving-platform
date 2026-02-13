"""
Integration tests for ML serving API
Tests actual API endpoints, not just unit tests
"""
import pytest
import requests
import time
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def base_url():
    """API base URL - change for different environments"""
    return "http://localhost:8000"


@pytest.fixture(scope="session")
def wait_for_api(base_url):
    """Wait for API to be ready"""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(1)
            else:
                raise
    raise RuntimeError("API did not become ready in time")


def test_root_endpoint(base_url, wait_for_api):
    """Test root endpoint returns service info"""
    response = requests.get(base_url)
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data


def test_health_endpoint(base_url, wait_for_api):
    """Test health check endpoint"""
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert "uptime_seconds" in data


def test_metrics_endpoint(base_url, wait_for_api):
    """Test Prometheus metrics endpoint"""
    response = requests.get(f"{base_url}/metrics")
    assert response.status_code == 200
    assert "model_predictions_total" in response.text
    assert "model_prediction_latency_seconds" in response.text


def test_valid_prediction(base_url, wait_for_api):
    """Test valid prediction request"""
    payload = {
        "features": [0.5] * 20,
        "request_id": "test_123"
    }
    
    response = requests.post(f"{base_url}/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
    assert data["model_version"] is not None
    assert data["request_id"] == "test_123"
    assert "latency_ms" in data
    assert data["latency_ms"] > 0


def test_prediction_invalid_features_count(base_url, wait_for_api):
    """Test prediction with wrong number of features"""
    payload = {
        "features": [0.5] * 10  # Wrong count
    }
    
    response = requests.post(f"{base_url}/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_prediction_invalid_features_nan(base_url, wait_for_api):
    """Test prediction with NaN values"""
    payload = {
        "features": [0.5] * 19 + [float('nan')]
    }
    
    response = requests.post(f"{base_url}/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_prediction_consistency(base_url, wait_for_api):
    """Test that same input gives same prediction"""
    payload = {
        "features": [0.8] * 20
    }
    
    # Make 5 predictions with same input
    predictions = []
    for _ in range(5):
        response = requests.post(f"{base_url}/predict", json=payload)
        assert response.status_code == 200
        predictions.append(response.json()["prediction"])
    
    # All predictions should be identical (deterministic model)
    assert len(set(predictions)) == 1


def test_prediction_performance(base_url, wait_for_api):
    """Test prediction latency is acceptable"""
    payload = {
        "features": [0.5] * 20
    }
    
    latencies = []
    for _ in range(10):
        start = time.time()
        response = requests.post(f"{base_url}/predict", json=payload)
        latency = (time.time() - start) * 1000
        
        assert response.status_code == 200
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    # Assertions on performance
    assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms"
    assert p95_latency < 200, f"P95 latency {p95_latency:.2f}ms exceeds 200ms"


def test_concurrent_requests(base_url, wait_for_api):
    """Test API handles concurrent requests"""
    import concurrent.futures
    
    def make_prediction():
        payload = {"features": [0.5] * 20}
        response = requests.post(f"{base_url}/predict", json=payload)
        return response.status_code
    
    # Make 20 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_prediction) for _ in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # All should succeed
    assert all(status == 200 for status in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
