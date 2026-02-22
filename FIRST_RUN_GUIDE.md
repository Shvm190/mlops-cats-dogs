# First Run Guide: `mlops-cats-dogs`

This guide walks through a clean first-time setup on macOS/Linux.

## 1. Prerequisites

Install these first:

- Python `3.10` or `3.11`
- `pip`
- `git`
- `curl`
- Optional: `docker` + `docker compose`
- Optional: `kubectl` (for Kubernetes deploy)
- Optional: `dvc` (already in `requirements.txt`)

## 2. Clone and Create a Virtual Environment

```bash
git clone <your-repo-url>
cd mlops-cats-dogs

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Prepare Data (Choose One)

### Option A: Pull tracked data with DVC

```bash
dvc pull
```

### Option B: Provide raw dataset and preprocess

Expected raw layout (either works):

```text
data/raw/PetImages/Cat/*.jpg
data/raw/PetImages/Dog/*.jpg
```

or:

```text
data/raw/cat/*.jpg
data/raw/dog/*.jpg
```

Then run:

```bash
python -m src.data.preprocess --raw-dir data/raw --processed-dir data/processed --image-size 224
```

## 4. Train the Model (First Artifact Generation)

This produces files required by the API under `artifacts/models/`.

```bash
python -m src.models.train --config configs/train_config.yaml
```

Note:
- The default config now uses local MLflow storage (`mlruns`) so no MLflow server is required for first run.
- If you want to force local tracking explicitly, run:

```bash
python -m src.models.train --config configs/train_config.yaml --mlflow-uri mlruns
```

- If you pass a remote MLflow URI (for example `http://127.0.0.1:5001`) and it is down,
  training now automatically falls back to local `mlruns` and continues.

Expected key outputs:

- `artifacts/models/model_torchscript.pt`
- `artifacts/models/model.pth`
- `artifacts/models/model_metadata.json`

## 5. Evaluate the Trained Model

```bash
python -m src.models.evaluate \
  --model-path artifacts/models/model_torchscript.pt \
  --test-dir data/processed/test \
  --output-dir artifacts/evaluation
```

## 6. Start the API Locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

In another terminal:

```bash
curl -s http://127.0.0.1:8080/health
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/ready
```

- `/health` gives service status (`healthy` or `degraded`)
- `/ready` must return `200` before inference traffic

## 7. Run a Prediction

```bash
curl -s -X POST "http://127.0.0.1:8080/predict" \
  -F "file=@path/to/your/image.jpg"
```

Or run the smoke test:

```bash
bash scripts/smoke_test.sh http://127.0.0.1:8080
```

Optional local UI:

```bash
API_BASE_URL=http://127.0.0.1:8080 streamlit run src/frontend/app.py --server.port 8501
```

## 8. Run with Docker

Build image:

```bash
docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .
```

Run container (mount local artifacts so model files are available):

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  cats-dogs-classifier:latest
```

## 9. Run Full Stack with Docker Compose

```bash
cd docker
docker compose up --build
```

Services:

- API: `http://127.0.0.1:8080` (override: `INFERENCE_HOST_PORT`)
- Frontend (Streamlit): `http://127.0.0.1:8501` (override: `FRONTEND_HOST_PORT`)
- MLflow: `http://127.0.0.1:5001` (override: `MLFLOW_HOST_PORT`)
- Prometheus: `http://127.0.0.1:9090` (override: `PROMETHEUS_HOST_PORT`)
- Grafana: `http://127.0.0.1:3000` (override: `GRAFANA_HOST_PORT`)
- Grafana login: `admin` / `admin123` (from compose env)

Example with custom host ports:

```bash
INFERENCE_HOST_PORT=8081 FRONTEND_HOST_PORT=8502 MLFLOW_HOST_PORT=5002 PROMETHEUS_HOST_PORT=9091 GRAFANA_HOST_PORT=3001 docker compose up --build
```

## 10. Run Tests

```bash
pytest tests/ -v
```

## 11. Common First-Run Issues

### `Model not found: artifacts/models/model_torchscript.pt`

Cause: model artifacts were not created or mounted.

Fix:

1. Re-run training with local MLflow URI:

```bash
python -m src.models.train --config configs/train_config.yaml --mlflow-uri mlruns
```

2. Ensure data splits exist and are non-empty:

```bash
for s in train val test; do
  echo "== $s =="
  ls "data/processed/$s/cat" 2>/dev/null | wc -l
  ls "data/processed/$s/dog" 2>/dev/null | wc -l
done
```

3. Then verify artifacts:

```bash
ls -lah artifacts/models
```

4. If using Docker, mount `artifacts/` into container exactly as shown.

### `ModuleNotFoundError: No module named 'src'`

Cause: running package files as scripts (`python src/...`) instead of module mode.

Fix (examples):

```bash
python -m src.models.train --config configs/train_config.yaml --mlflow-uri mlruns
python -m src.models.evaluate --model-path artifacts/models/model_torchscript.pt --test-dir data/processed/test --output-dir artifacts/evaluation
```

### `No module named 'structlog'` when starting API

Cause: env missing dependencies.

Fix:

```bash
pip install -r requirements.txt
```

### `/predict` returns `Expected UploadFile, received: <class 'str'>`

Cause: curl form upload missing `@` before path.

Fix:

```bash
curl -s -X POST "http://127.0.0.1:8080/predict" \
  -F "file=@data/processed/test/dog/3.jpg"
```

### Docker Compose conflict: container name already in use

Fix:

```bash
cd docker
docker compose down --remove-orphans
docker ps -a --filter name=prometheus
docker rm -f <container-name-if-needed>
docker compose up --build
```

### Docker Compose port already allocated (`5000`, `8080`, etc.)

Check who owns the port:

```bash
lsof -i tcp:5000
lsof -i tcp:8080
```

Run compose with different host ports:

```bash
INFERENCE_HOST_PORT=8081 FRONTEND_HOST_PORT=8502 MLFLOW_HOST_PORT=5002 PROMETHEUS_HOST_PORT=9091 GRAFANA_HOST_PORT=3001 docker compose up --build
```

### Docker build fails with `failed to commit snapshot ... input/output error`

Usually Docker Desktop disk/cache corruption or low space.

Fix sequence:

```bash
docker system df
docker builder prune -af
docker system prune -af --volumes
```

Then restart Docker Desktop and rebuild:

```bash
cd /Users/homeaccount/PycharmProjects/mlops-cats-dogs
docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .
```

Note: a root `.dockerignore` is included to keep build context small (`data/`, `artifacts/`, `mlruns/` are excluded).

### MLflow UI opens but shows no runs

Cause: training was logged to local filesystem (`mlruns`) while Docker MLflow uses its own container volume.

Fix:

1. Check MLflow host port:

```bash
cd /Users/homeaccount/PycharmProjects/mlops-cats-dogs/docker
docker compose -p docker ps
```

2. Trigger one short run and log directly to Docker MLflow (replace `5001` if your mapped port is different):

```bash
cd /Users/homeaccount/PycharmProjects/mlops-cats-dogs
python -m src.models.train --config configs/train_config.yaml --epochs 1 --mlflow-uri http://127.0.0.1:5001
```

3. Refresh MLflow UI and open experiment `cats-dogs-binary-classification`.

### MLflow artifact error: `OSError: [Errno 30] Read-only file system: '/mlflow'`

Cause: MLflow tracking server started without artifact-serving mode, so client tries to write artifacts to local `file:///mlflow/...`.

Fix:

```bash
cd /Users/homeaccount/PycharmProjects/mlops-cats-dogs/docker
docker compose up -d --force-recreate mlflow
```

Then retry training with:

```bash
cd /Users/homeaccount/PycharmProjects/mlops-cats-dogs
python -m src.models.train --config configs/train_config.yaml --epochs 1 --mlflow-uri http://127.0.0.1:5001
```

### API is up but `/ready` returns `503`

Cause: model failed to load.

Fix:

1. Check `MODEL_PATH` and `METADATA_PATH` env vars.
2. Confirm files exist under `artifacts/models/`.
3. Inspect API logs for the exact loading error.

### OpenMP runtime error during tests (`Can't open SHM2`)

If your environment has shared-memory restrictions, try:

```bash
OMP_NUM_THREADS=1 pytest tests/ -v
```

If needed for local debugging only:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest tests/ -v
```

## 12. First Deploy Checkpoint

Before first deploy, verify all are true:

- `/health` returns JSON with `"model_loaded": true`
- `/ready` returns HTTP `200`
- `bash scripts/smoke_test.sh <base-url>` passes
- `artifacts/models/model_torchscript.pt` and `model_metadata.json` exist

## 13. Requirement Clarifications

### M2 API framework used

This project uses **FastAPI** (not Flask):

- API app and routes: `src/api/main.py`
- Optional UI: **Streamlit** frontend at `src/frontend/app.py` (calls FastAPI `/predict`)

### Monitoring metrics tracked (and why they matter)

The service exposes metrics that are meaningful for inference monitoring:

- `http_requests_total` by method/path: request traffic and endpoint load.
- `predictions_total` by label: class distribution drift over time.
- `prediction_latency_ms` histogram: P50/P95/P99 latency SLO tracking.
- `prediction_errors_total` by error type: reliability and failure triage.
- `model_loaded` gauge: readiness state of inference model.

These metrics are available on `/metrics`, scraped by Prometheus, and visualized in Grafana.

### CD behavior: does deployment train a new model?

Current CD **does not train** a model during deploy.

- CI/CD builds and deploys the container image.
- Inference uses model artifacts already present at runtime (`/app/artifacts/...`).
- For Docker Compose, artifacts come from local mount: `../artifacts:/app/artifacts:ro`.
- For Kubernetes, artifacts are expected via mounted PVC (`model-artifacts-pvc`).

If you want training inside CI, add a dedicated training stage that publishes artifacts before deploy.

## 14. Deploy-Mode Validation (API + Monitoring)

### A. Docker Compose deployment validation

1. Start stack:

```bash
cd docker
docker compose up -d --build
```

2. Health/readiness/predict checks:

```bash
curl -s http://127.0.0.1:8080/health
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/ready
curl -s -X POST "http://127.0.0.1:8080/predict" -F "file=@../data/processed/test/dog/3.jpg"
```

Open Streamlit UI for interactive image upload/prediction:

```bash
open http://127.0.0.1:8501
```

3. Verify metrics are changing:

```bash
curl -s http://127.0.0.1:8080/metrics | grep -E "http_requests_total|predictions_total|prediction_latency_ms|prediction_errors_total"
curl -s "http://127.0.0.1:9090/api/v1/query?query=sum(predictions_total)"
curl -s "http://127.0.0.1:9090/api/v1/query?query=histogram_quantile(0.95,rate(prediction_latency_ms_bucket[5m]))"
```

4. Grafana dashboard:

- URL: `http://127.0.0.1:3000`
- Username: `admin`
- Password: `admin123`
- Open dashboard: `Cats vs Dogs — Model Inference Dashboard`

### B. Kubernetes deployment validation

After deploy, port-forward service and run checks:

```bash
kubectl -n mlops port-forward svc/cats-dogs-classifier-svc 18080:80
```

Then in another terminal:

```bash
curl -s http://127.0.0.1:18080/health
curl -s -X POST "http://127.0.0.1:18080/predict" -F "file=@data/processed/test/dog/3.jpg"
bash scripts/smoke_test.sh http://127.0.0.1:18080
```

## 15. Completion Steps for Your Submission

Run these to close remaining rubric gaps and package deliverables.

### 1) Ensure DVC reproducibility state is materialized

```bash
cd /Users/homeaccount/PycharmProjects/mlops-cats-dogs
dvc repro
git add dvc.yaml dvc.lock .dvc/config
```

### 2) Replace template placeholders in deployment manifests

Before final submission, replace `YOUR_ORG` and repository placeholders:

```bash
cd /Users/homeaccount/PycharmProjects/mlops-cats-dogs
sed -i '' 's|YOUR_ORG|<your-github-org-or-user>|g' k8s/manifests.yaml k8s/argocd-application.yaml .github/workflows/ci-cd.yml
sed -i '' 's|https://github.com/YOUR_ORG/mlops-cats-dogs.git|https://github.com/<your-github-org-or-user>/mlops-cats-dogs.git|g' k8s/argocd-application.yaml
```

### 3) Ensure trained artifacts are present

```bash
ls -lah artifacts/models
ls -lah artifacts/evaluation
```

### 4) Optional: create a loss-curve image from training history

If reviewer expects an explicit curve image artifact:

```bash
python -m pip install matplotlib==3.8.3
python - <<'PY'
import json
import matplotlib.pyplot as plt
from pathlib import Path

p = Path("artifacts/models/training_history.json")
h = json.loads(p.read_text())
plt.figure(figsize=(8,4))
plt.plot(h["train_loss"], label="train_loss")
plt.plot(h["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
out = Path("artifacts/models/loss_curve.png")
plt.savefig(out, dpi=150)
print(out)
PY
```

### 5) Zip deliverable package

```bash
cd /Users/homeaccount/PycharmProjects/mlops-cats-dogs
zip -r mlops-cats-dogs-deliverable.zip \
  src tests docker k8s scripts configs \
  .github/workflows dvc.yaml .dvc requirements.txt pyproject.toml \
  MLproject README.md FIRST_RUN_GUIDE.md \
  artifacts/models artifacts/evaluation
```

### 6) 5-minute recording flow (recommended script)

Record these steps in sequence:

1. `0:00-0:30` Show project structure and milestone mapping (M1-M5).
2. `0:30-1:00` Show versioning evidence: `git log --oneline -n 3` and `dvc status`.
3. `1:00-1:40` Run short training to MLflow server:
   `python -m src.models.train --config configs/train_config.yaml --epochs 1 --mlflow-uri http://127.0.0.1:5001`
4. `1:40-2:10` Show `artifacts/models` and MLflow run (params, metrics, artifacts).
5. `2:10-2:40` Start deployment:
   `cd docker && docker compose -p docker up -d --build`
6. `2:40-3:20` Verify API:
   `/health`, `/ready`, and one `/predict` curl.
7. `3:20-3:50` Open Streamlit and run one interactive prediction.
8. `3:50-4:30` Show monitoring:
   `/metrics` output, Prometheus query `sum(predictions_total)`, Grafana dashboard.
9. `4:30-5:00` Show CI/CD files and zip command for final deliverable.
