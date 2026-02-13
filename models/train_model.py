"""
Fraud Detection Model Training
Demonstrates production ML practices: versioning, metrics tracking, artifact management
"""
import pickle
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def generate_synthetic_data(n_samples=10000):
    """Generate synthetic fraud detection data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],  # Imbalanced like real fraud data
        flip_y=0.01,
        random_state=42
    )
    return X, y


def train_model(X_train, y_train):
    """Train fraud detection model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics


def save_model_artifacts(model, metrics, version):
    """Save model with version and metadata"""
    model_dir = Path(__file__).parent / f"v{version}"
    model_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": "RandomForestClassifier",
        "n_features": 20,
        "metrics": metrics
    }
    
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create/update latest symlink
    latest_link = Path(__file__).parent / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(f"v{version}", target_is_directory=True)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    print(f"✅ Latest version: v{version}")
    print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")


def main():
    print("🚀 Training fraud detection model...")
    
    # Generate data
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"📊 Fraud rate: {y_train.mean():.2%}")
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save with version
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_model_artifacts(model, metrics, version)


if __name__ == "__main__":
    main()