"""
Unit tests for model training and API logic
These tests don't require a running API server
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.train_model import generate_synthetic_data, train_model, evaluate_model


def test_generate_synthetic_data():
    """Test that synthetic data generation works"""
    X, y = generate_synthetic_data(n_samples=1000)
    
    assert X.shape == (1000, 20)
    assert y.shape == (1000,)
    assert set(y) == {0, 1}  # Binary classification
    assert 0.03 < y.mean() < 0.07  # Should be ~5% fraud


def test_train_model():
    """Test that model training works"""
    X, y = generate_synthetic_data(n_samples=1000)
    
    model = train_model(X, y)
    
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    # Test prediction
    predictions = model.predict(X[:10])
    assert predictions.shape == (10,)
    assert all(p in [0, 1] for p in predictions)


def test_evaluate_model():
    """Test that model evaluation works"""
    X, y = generate_synthetic_data(n_samples=1000)
    model = train_model(X, y)
    
    metrics = evaluate_model(model, X, y)
    
    assert 'roc_auc' in metrics
    assert 'classification_report' in metrics
    assert 0.5 < metrics['roc_auc'] < 1.0  # Should be better than random


def test_model_prediction_shapes():
    """Test that predictions have correct shapes"""
    X, y = generate_synthetic_data(n_samples=100)
    model = train_model(X, y)
    
    # Single prediction
    single_pred = model.predict(X[0:1])
    assert single_pred.shape == (1,)
    
    # Batch prediction
    batch_pred = model.predict(X[:10])
    assert batch_pred.shape == (10,)
    
    # Probability prediction
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 2)


def test_feature_validation():
    """Test that features are validated correctly"""
    from app.main import PredictionRequest
    from pydantic import ValidationError
    
    # Valid request
    valid = PredictionRequest(features=[0.5] * 20)
    assert len(valid.features) == 20
    
    # Too few features
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.5] * 10)
    
    # Too many features
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.5] * 30)
    
    # NaN values
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.5] * 19 + [float('nan')])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
