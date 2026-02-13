"""
ML Model Serving API
Production-ready FastAPI app with metrics, logging, and health checks
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['model_version', 'status']
)
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Model prediction latency',
    ['model_version']
)
MODEL_LOAD_TIME = Gauge(
    'model_load_timestamp',
    'Timestamp when model was loaded'
)
FRAUD_PREDICTIONS = Counter(
    'fraud_predictions_total',
    'Total fraud predictions (positive class)',
    ['model_version']
)

# Request/Response models
class PredictionRequest(BaseModel):
    """Request schema for predictions"""
    features: List[float] = Field(..., min_items=20, max_items=20)
    request_id: str = Field(default=None)
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features contain NaN or Inf values")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [0.1] * 20,
                "request_id": "req_123"
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: int
    probability: float
    model_version: str
    request_id: str = None
    latency_ms: float
    
    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str = None
    uptime_seconds: float
    
    model_config = {"protected_namespaces": ()}


# Model loader
class ModelLoader:
    """Handles model loading with versioning"""
    
    def __init__(self, model_path: Path = None):
        self.model = None
        self.metadata = None
        
        # Default path logic: try /app/models/latest (Docker) or ../models/latest (local)
        if model_path is None:
            docker_path = Path("/app/models/latest")
            local_path = Path(__file__).parent.parent / "models" / "latest"
            
            if docker_path.exists():
                self.model_path = docker_path
            elif local_path.exists():
                self.model_path = local_path
            else:
                # If neither exists, use local path (will fail with clear error)
                self.model_path = local_path
        else:
            self.model_path = model_path
            
        self.load_model()
    
    def load_model(self):
        """Load model and metadata"""
        try:
            model_file = self.model_path / "model.pkl"
            metadata_file = self.model_path / "metadata.json"
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            self.model = joblib.load(model_file)
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"version": "unknown"}
            
            MODEL_LOAD_TIME.set_to_current_time()
            logger.info(f"✅ Model loaded: version={self.metadata.get('version')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction with timing"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            prediction = int(self.model.predict(features)[0])
            probability = float(self.model.predict_proba(features)[0][1])
            
            latency = (time.time() - start_time) * 1000  # ms
            
            return {
                "prediction": prediction,
                "probability": probability,
                "latency_ms": latency
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving Platform",
    description="Production-ready ML model serving with monitoring",
    version="1.0.0"
)

# Global model loader
model_loader = None
app_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_loader
    try:
        model_loader = ModelLoader()
        logger.info("🚀 Application startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.2f}ms"
    )
    
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for K8s liveness/readiness probes"""
    return HealthResponse(
        status="healthy" if model_loader and model_loader.model else "unhealthy",
        model_loaded=model_loader is not None and model_loader.model is not None,
        model_version=model_loader.metadata.get("version") if model_loader else None,
        uptime_seconds=time.time() - app_start_time
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make fraud prediction
    
    Returns prediction (0=legitimate, 1=fraud) with probability
    """
    if model_loader is None or model_loader.model is None:
        PREDICTION_COUNTER.labels(
            model_version="unknown",
            status="error"
        ).inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Predict with timing
        with PREDICTION_LATENCY.labels(
            model_version=model_loader.metadata.get("version", "unknown")
        ).time():
            result = model_loader.predict(features)
        
        # Update metrics
        model_version = model_loader.metadata.get("version", "unknown")
        PREDICTION_COUNTER.labels(
            model_version=model_version,
            status="success"
        ).inc()
        
        if result["prediction"] == 1:
            FRAUD_PREDICTIONS.labels(model_version=model_version).inc()
        
        # Build response
        response = PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            model_version=model_version,
            request_id=request.request_id,
            latency_ms=result["latency_ms"]
        )
        
        logger.info(
            f"Prediction: {result['prediction']} "
            f"(p={result['probability']:.4f}, "
            f"latency={result['latency_ms']:.2f}ms)"
        )
        
        return response
        
    except ValueError as e:
        PREDICTION_COUNTER.labels(
            model_version=model_loader.metadata.get("version", "unknown"),
            status="validation_error"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        PREDICTION_COUNTER.labels(
            model_version=model_loader.metadata.get("version", "unknown"),
            status="error"
        ).inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "ML Model Serving Platform",
        "version": "1.0.0",
        "model_version": model_loader.metadata.get("version") if model_loader else None,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)