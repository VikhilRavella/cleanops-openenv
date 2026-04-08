# ── CleanOps OpenEnv — Dockerfile ────────────────────────────────────────────
# Compatible with Hugging Face Spaces (port 7860)
# Build: docker build -t cleanops-openenv -f server/Dockerfile .
# Run:   docker run -p 7860:7860 cleanops-openenv

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="CleanOps OpenEnv"
LABEL org.opencontainers.image.description="AI agent environment for data cleaning benchmarks"
LABEL org.opencontainers.image.version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
