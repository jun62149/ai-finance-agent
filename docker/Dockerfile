# AI 金融智能投研 Agent — API + Web UI image (CPU default).
#
# Serves the Quant API, the RAG retrieval endpoints, the Agent (/api/chat) and the
# single-page UI at "/". Qdrant runs as a separate service (see docker-compose.yml).
#
# Build (from the repository root):
#   docker build -f docker/Dockerfile -t ai-finance-agent:latest .
# Run:
#   docker run --rm -p 8000:8000 --env-file .env \
#     -e QDRANT_URL=http://host.docker.internal:6333 \
#     -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
#     ai-finance-agent:latest
#
# Notes
# - `data/` (SQLite + BM25 index) and `models/` (XGBoost pkl) are mounted rather
#   than baked in: they are large, gitignored, and rebuilt by the scripts.
# - CPU-only torch keeps the image and the download reasonable. Use
#   Dockerfile.gpu (with `--gpus all`) when the reranker/embedder must run on GPU.
# - EMBEDDING_MODEL / RERANKER_MODEL are downloaded from HuggingFace on first use;
#   mount a cache (see compose) or set HF_ENDPOINT=https://hf-mirror.com in China.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Mirrors: the defaults target Chinese networks (deb.debian.org and PyPI are slow
# or unreachable there). Override with --build-arg for a neutral build:
#   docker build --build-arg APT_MIRROR=deb.debian.org #                --build-arg PIP_INDEX_URL=https://pypi.org/simple ...
ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN set -eux; \
    if [ -n "${APT_MIRROR}" ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list 2>/dev/null || true; \
      find /etc/apt/sources.list.d -name '*.sources' -exec sed -i "s|deb.debian.org|${APT_MIRROR}|g" {} + 2>/dev/null || true; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl libgomp1; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# torch first, from the CPU wheel index. `--extra-index-url` lets transitive
# dependencies come from the (fast, regional) PyPI mirror; there is deliberately
# no `||` fallback, because falling back to PyPI would silently install the
# CUDA-enabled torch plus ~4GB of nvidia-* wheels into a CPU image.
RUN pip install --no-cache-dir --timeout 60 --retries 5 \
      --index-url "${TORCH_INDEX_URL}" \
      --extra-index-url "${PIP_INDEX_URL}" \
      torch
RUN pip install --no-cache-dir --timeout 60 --retries 5 -r requirements.txt -i "${PIP_INDEX_URL}"

COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY tests ./tests

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
