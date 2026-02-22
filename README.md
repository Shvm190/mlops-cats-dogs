# 🐱🐶 MLOps Pipeline: Cats vs Dogs Classification

A production-grade, end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) built for a pet adoption platform.

## First-Time Setup

- Detailed onboarding guide: [`FIRST_RUN_GUIDE.md`](FIRST_RUN_GUIDE.md)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                               │
│                                                                     │
│  Data → Preprocess → Train → Track → Package → CI → CD → Monitor  │
│                                                                     │
│  DVC    DVC/Python   PyTorch  MLflow  FastAPI  GH     K8s/Docker  Prometheus│
└─────────────────────────────────────────────────────────────────────┘
```

## Milestones

| Milestone | Description | Tools |
|-----------|-------------|-------|
| M1 | Model Development & Experiment Tracking | Git, DVC, PyTorch, MLflow |
| M2 | Model Packaging & Containerization | FastAPI, Docker |
| M3 | CI Pipeline | GitHub Actions, pytest |
| M4 | CD Pipeline & Deployment | Kubernetes/Docker Compose, Argo CD |
| M5 | Monitoring & Logging | Prometheus, Grafana, structured logging |

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> && cd mlops-cats-dogs
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Pull data with DVC
dvc pull

# 3. Preprocess data
python -m src.data.preprocess

# 4. Train model
python -m src.models.train --config configs/train_config.yaml --mlflow-uri mlruns

# 5. Run inference service locally
uvicorn src.api.main:app --host 0.0.0.0 --port 8080

# 5b. (Optional) Run Streamlit frontend locally
API_BASE_URL=http://127.0.0.1:8080 streamlit run src/frontend/app.py --server.port 8501

# 6. Docker build & run
docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .
docker run -p 8080:8080 -v $(pwd)/artifacts:/app/artifacts:ro cats-dogs-classifier:latest

# 7. Run tests
pytest tests/ -v

# 8. Kubernetes deploy
kubectl apply -f k8s/
```

## Project Structure

```
mlops-cats-dogs/
├── configs/               # Training & app configs
├── data/
│   ├── raw/               # Raw Kaggle dataset (DVC tracked)
│   └── processed/         # Preprocessed 224x224 images (DVC tracked)
├── docker/                # Dockerfile and compose files
├── k8s/                   # Kubernetes manifests
├── notebooks/             # EDA and experimentation notebooks
├── scripts/               # Utility scripts (smoke tests, monitoring)
├── src/
│   ├── api/               # FastAPI inference service
│   ├── frontend/          # Streamlit UI for image upload/prediction
│   ├── data/              # Data preprocessing pipeline
│   ├── models/            # Model architecture, training, evaluation
│   └── monitoring/        # Metrics collection and logging
├── tests/                 # Unit and integration tests
├── .github/workflows/     # CI/CD GitHub Actions
├── .dvc/                  # DVC configuration
├── dvc.yaml               # DVC pipeline stages
├── requirements.txt       # Python dependencies
└── MLproject              # MLflow project file
```

## Dataset

- **Source**: [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Preprocessing**: Resized to 224×224 RGB, normalized
- **Splits**: 80% train / 10% validation / 10% test
- **Augmentation**: RandomHorizontalFlip, RandomRotation, ColorJitter

## Model

- **Architecture**: MobileNetV2 (fine-tuned) + baseline simple CNN
- **Framework**: PyTorch
- **Format**: `.pt` (TorchScript for production, `.pth` for checkpoints)
- **Target Metrics**: >90% accuracy on test set

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/ready` | GET | Readiness check (model loaded) |
| `/predict` | POST | Predict cat or dog from image |
| `/metrics` | GET | Prometheus metrics |
| `/model/info` | GET | Model metadata and version |

## Optional Frontend

- Streamlit app: `src/frontend/app.py`
- Local run:

```bash
API_BASE_URL=http://127.0.0.1:8080 streamlit run src/frontend/app.py --server.port 8501
```
