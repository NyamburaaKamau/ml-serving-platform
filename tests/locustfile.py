"""
Load testing for ML serving API
Run with: locust -f tests/locustfile.py --host=http://localhost:8000
"""
from locust import HttpUser, task, between
import random


class MLServingUser(HttpUser):
    """Simulates user making predictions"""
    
    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5s between requests
    
    def on_start(self):
        """Called when user starts - check API is ready"""
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("API not healthy")
    
    @task(10)  # Weight: 10 (most common task)
    def predict(self):
        """Make a prediction request"""
        features = [random.uniform(0, 1) for _ in range(20)]
        payload = {
            "features": features,
            "request_id": f"load_test_{random.randint(1000, 9999)}"
        }
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response
                if "prediction" not in data or "probability" not in data:
                    response.failure("Invalid response format")
                elif data["latency_ms"] > 1000:
                    response.failure(f"High latency: {data['latency_ms']}ms")
                else:
                    response.success()
            else:
                response.failure(f"Got status {response.status_code}")
    
    @task(2)  # Weight: 2 (occasional health checks)
    def health_check(self):
        """Check health endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("API unhealthy")
            else:
                response.failure(f"Got status {response.status_code}")
    
    @task(1)  # Weight: 1 (rare metrics checks)
    def metrics_check(self):
        """Check metrics endpoint"""
        self.client.get("/metrics")
