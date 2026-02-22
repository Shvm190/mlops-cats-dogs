# ============================================================
# Dockerfile — Cats vs Dogs Inference Service
# Multi-stage build for minimal production image
# ============================================================

# ─── Stage 1: Builder ────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python runtime dependencies
COPY requirements-runtime.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user --no-warn-script-location -r requirements-runtime.txt


# ─── Stage 2: Production ─────────────────────────────────────
FROM python:3.10-slim AS production

# Metadata
LABEL maintainer="mlops-team@petadoption.com"
LABEL version="1.0.0"
LABEL description="Cats vs Dogs binary image classifier — inference service"

WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/sh --create-home appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH="/home/appuser/.local/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for model artifacts and logs
RUN mkdir -p artifacts/models artifacts/checkpoints logs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default environment variables
ENV MODEL_PATH="artifacts/models/model_torchscript.pt"
ENV METADATA_PATH="artifacts/models/model_metadata.json"
ENV MAX_UPLOAD_SIZE_MB="10"
ENV LOG_LEVEL="INFO"
ENV HOST="0.0.0.0"
ENV PORT="8080"
ENV WORKERS="1"

# Start server
CMD ["sh", "-c", "uvicorn src.api.main:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
