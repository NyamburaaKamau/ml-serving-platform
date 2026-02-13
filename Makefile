.PHONY: help install train test docker-build k8s-deploy clean

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install Python dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov requests locust

train: ## Train the ML model
	python models/train_model.py

run: ## Run the API locally
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests with coverage
	pytest tests/ -v --cov=app --cov-report=term-missing

test-api: ## Run API integration tests
	pytest tests/test_api.py -v

load-test: ## Run load tests (requires running API)
	@echo "Starting load test..."
	@echo "Open http://localhost:8089 in your browser"
	locust -f tests/locustfile.py --host=http://localhost:8000

docker-build: ## Build Docker image
	docker build -t ml-serving-platform:latest .

docker-run: ## Run Docker container locally
	docker run -p 8000:8000 ml-serving-platform:latest

docker-test: docker-build ## Build and test Docker image
	docker run -d -p 8000:8000 --name ml-serving-test ml-serving-platform:latest
	sleep 5
	curl http://localhost:8000/health
	docker stop ml-serving-test
	docker rm ml-serving-test

k8s-deploy: ## Deploy to local Kubernetes
	kubectl apply -f k8s/

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -f k8s/

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods,svc,hpa

k8s-logs: ## Tail logs from all pods
	kubectl logs -l app=ml-serving -f --tail=100

k8s-port-forward: ## Forward port to access service locally
	kubectl port-forward svc/ml-serving 8000:80

minikube-start: ## Start minikube cluster
	minikube start --cpus=4 --memory=8192

minikube-load: docker-build ## Build and load image to minikube
	minikube image load ml-serving-platform:latest

minikube-url: ## Get minikube service URL
	minikube service ml-serving --url

clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov

format: ## Format code with black
	black app/ models/ tests/

lint: ## Lint code with ruff
	ruff check app/ models/ tests/
