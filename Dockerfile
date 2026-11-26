# Multi-stage, non-root Dockerfile for ProbCount
FROM python:3.13-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install ".[api,dashboard]"

FROM python:3.13-slim AS runtime

RUN useradd --create-home --shell /bin/bash probcount
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/
COPY dashboard/ dashboard/
COPY scripts/ scripts/

RUN mkdir -p /app/data/snapshots && chown -R probcount:probcount /app
USER probcount

EXPOSE 8000 8501
CMD ["uvicorn", "probcount.api:app", "--host", "0.0.0.0", "--port", "8000"]
