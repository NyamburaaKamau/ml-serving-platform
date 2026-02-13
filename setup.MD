# 🚀 SETUP GUIDE - Run This On Your Machine

## What We Built Today

A **production-grade ML serving platform** that demonstrates you understand real infrastructure, not bootcamps.

This shows hiring managers you can:
- Deploy ML models to production
- Build scalable infrastructure with Kubernetes
- Implement proper monitoring and observability
- Write production-ready code with error handling
- Think about cost, security, and reliability

---

## 📋 Prerequisites

Install these on your machine:
```bash
# Python 3.11+
python --version

# Docker
docker --version

# Kubernetes (choose one):
# Option 1: minikube (easiest for local)
minikube version

# Option 2: kind
kind version

# Option 3: Docker Desktop (has built-in K8s)
```

---

## ⚡ Quick Start (15 minutes)

### Step 1: Get the code
```bash
# Download/clone this project
cd ml-serving-platform
```

### Step 2: Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov requests
```

### Step 3: Train the model
```bash
python models/train_model.py
```

**Expected output:**
```
🚀 Training fraud detection model...
📊 Training samples: 8000, Test samples: 2000
📊 Fraud rate: 5.00%
✅ Model saved: models/v20250212_160000/model.pkl
✅ Metadata saved: models/v20250212_160000/metadata.json
✅ Latest version: v20250212_160000
📊 ROC-AUC: 0.9845
```

### Step 4: Run the API locally
```bash
uvicorn app.main:app --reload
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
✅ Model loaded: version=v20250212_160000
🚀 Application startup complete
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 5: Test it!

**Open another terminal** and test the endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.5,
                 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.5, 0.3],
    "request_id": "test_001"
  }'

# View metrics
curl http://localhost:8000/metrics

# Interactive API docs
open http://localhost:8000/docs
```

**Expected prediction response:**
```json
{
  "prediction": 0,
  "probability": 0.05,
  "model_version": "v20250212_160000",
  "request_id": "test_001",
  "latency_ms": 12.34
}
```

### Step 6: Run the tests
```bash
# Make sure API is running first!
pytest tests/test_api.py -v

# With coverage
pytest tests/ -v --cov=app --cov-report=html
open htmlcov/index.html  # View coverage report
```

---

## 🐳 Docker Testing (Next 10 minutes)

### Step 1: Build the Docker image
```bash
docker build -t ml-serving-platform:latest .
```

This will:
- Use multi-stage build to minimize image size
- Install dependencies in isolated environment
- Copy your trained model
- Set up non-root user for security
- Configure health checks

### Step 2: Run the container
```bash
docker run -p 8000:8000 ml-serving-platform:latest
```

### Step 3: Test the containerized API
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5] * 20}'
```

---

## ☸️ Kubernetes Deployment (Next 20 minutes)

### Step 1: Start local Kubernetes cluster

**Using minikube:**
```bash
minikube start --cpus=4 --memory=8192
```

**Using kind:**
```bash
kind create cluster --name ml-serving
```

**Using Docker Desktop:**
- Go to Settings → Kubernetes → Enable Kubernetes

### Step 2: Build and load the image

**For minikube:**
```bash
eval $(minikube docker-env)  # Use minikube's Docker daemon
docker build -t ml-serving-platform:latest .
```

**For kind:**
```bash
docker build -t ml-serving-platform:latest .
kind load docker-image ml-serving-platform:latest --name ml-serving
```

**For Docker Desktop:**
```bash
docker build -t ml-serving-platform:latest .
```

### Step 3: Deploy to Kubernetes
```bash
kubectl apply -f k8s/
```

**Expected output:**
```
deployment.apps/ml-serving created
service/ml-serving created
horizontalpodautoscaler.autoscaling/ml-serving-hpa created
```

### Step 4: Verify deployment
```bash
# Check pods
kubectl get pods

# Should see something like:
# NAME                          READY   STATUS    RESTARTS   AGE
# ml-serving-5d4b7c9f8d-abc12   1/1     Running   0          30s
# ml-serving-5d4b7c9f8d-xyz89   1/1     Running   0          30s

# Check service
kubectl get svc

# Check autoscaler
kubectl get hpa
```

### Step 5: Access the service

**For minikube:**
```bash
minikube service ml-serving --url
# Use the URL it gives you
```

**For kind/Docker Desktop:**
```bash
kubectl port-forward svc/ml-serving 8000:80
# Then use http://localhost:8000
```

### Step 6: Test the deployed service
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.5, 0.3]}'
```

### Step 7: Test autoscaling
```bash
# Generate load
kubectl run -it load-generator --rm --image=busybox --restart=Never -- \
  /bin/sh -c "while sleep 0.01; do wget -q -O- http://ml-serving/predict; done"

# In another terminal, watch it scale
kubectl get hpa ml-serving-hpa --watch

# You'll see:
# NAME              REFERENCE               TARGETS         MINPODS   MAXPODS   REPLICAS   AGE
# ml-serving-hpa    Deployment/ml-serving   45%/70%         2         10        2          2m
# ml-serving-hpa    Deployment/ml-serving   78%/70%         2         10        3          3m  <- Scaled up!
```

### Step 8: Check logs
```bash
# View logs from all pods
kubectl logs -l app=ml-serving --tail=50

# Follow logs in real-time
kubectl logs -l app=ml-serving -f

# Logs from specific pod
kubectl logs ml-serving-5d4b7c9f8d-abc12
```

---

## 🎯 What to Do Next

### For Your GitHub Profile:

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "feat: production ML serving platform with K8s"
git remote add origin https://github.com/YOUR_USERNAME/ml-serving-platform.git
git push -u origin main
```

2. **Pin this repo** on your GitHub profile

3. **Add topics** to the repo:
   - `mlops`
   - `kubernetes`
   - `platform-engineering`
   - `fastapi`
   - `production-ml`
   - `docker`
   - `prometheus`

4. **Create a demo GIF:**
   - Record deploying to K8s
   - Show the autoscaling in action
   - Add to README

### For Your LinkedIn:

Post about building this. Example:

```
Just built a production-grade ML serving platform from scratch 🚀

Not a bootcamp project. A real production system with:
• Kubernetes deployment with auto-scaling
• Zero-downtime rolling updates
• Prometheus monitoring & health checks
• CI/CD pipeline with security scanning
• Production-ready Docker containers
• Comprehensive integration tests

The difference? This solves actual production problems:
- Models that stay up during deployments
- Infrastructure that scales under load
- Observability that helps debug issues
- Cost optimization through resource limits

Tech stack: Python • FastAPI • Docker • Kubernetes • Prometheus

Check it out: [GitHub link]

#MLOps #Kubernetes #PlatformEngineering #Python
```

### Improvements to Add:

**Week 1:**
- [ ] Add Grafana dashboard for visualization
- [ ] Implement model A/B testing (two deployments)
- [ ] Add Terraform for cloud deployment

**Week 2:**
- [ ] Implement model drift detection
- [ ] Add data quality checks
- [ ] Create custom metrics dashboard

**Week 3:**
- [ ] Implement canary deployments
- [ ] Add distributed tracing (OpenTelemetry)
- [ ] Deploy to actual cloud (AWS/GCP/Azure)

---

## 🐛 Troubleshooting

### Model not loading
```bash
# Check if model exists
ls -la models/latest/

# Should see:
# model.pkl
# metadata.json

# If not, train again:
python models/train_model.py
```

### Docker build fails
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t ml-serving-platform:latest .
```

### Kubernetes pods not starting
```bash
# Check pod status
kubectl describe pod <pod-name>

# Common issues:
# 1. Image not found → Load image to cluster
# 2. Resource limits too low → Increase in deployment.yaml
# 3. Model not in image → Rebuild Docker image after training
```

### Tests failing
```bash
# Make sure API is running
curl http://localhost:8000/health

# Check if dependencies installed
pip list | grep -E "pytest|requests"

# Run with verbose output
pytest tests/test_api.py -v -s
```

---

## 📚 Understanding the Code

Read the code in this order:
1. `models/train_model.py` - See how models are versioned
2. `app/main.py` - Understand the API structure
3. `k8s/deployment.yaml` - Learn Kubernetes concepts
4. `.github/workflows/ci-cd.yaml` - See the CI/CD pipeline

---

## 💬 Questions?

This is a learning project. If something doesn't work:
1. Check the error messages carefully
2. Read the relevant documentation
3. Try the troubleshooting steps above
4. Google the error (that's what senior engineers do!)

---

**You built this today. Now ship it. 🚀**
