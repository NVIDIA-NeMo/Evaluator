# NeMo Evaluator base image.
# Contains: nemo-evaluator[scoring,stats] + built-in BYOB benchmarks.
# Sufficient for: built-in benchmarks, gym://, pi://, container://.
#
# Build:
#   docker build -t nemo-evaluator:0.7.0 .
#
# Per-harness variants:
#   docker build -f docker/Dockerfile.lm-eval -t nemo-evaluator:0.7.0-lm-eval .
#   docker build -f docker/Dockerfile.skills  -t nemo-evaluator:0.7.0-skills  .
#   docker build -f docker/Dockerfile.mteb    -t nemo-evaluator:0.7.0-mteb    .
#   docker build -f docker/Dockerfile.full    -t nemo-evaluator:0.7.0-full    .

FROM python:3.11-slim AS base

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md ./
COPY src/ src/
COPY benchmarks/ benchmarks/

RUN pip install --no-cache-dir ".[scoring,stats]"

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["nel"]
CMD ["--help"]
