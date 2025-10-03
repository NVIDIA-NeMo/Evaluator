#!/bin/bash
# Pull pre-built evaluation containers from NVIDIA NGC

# Set container version (or use environment variable)
DOCKER_TAG="${DOCKER_TAG:-25.08.1}"

# [snippet-start]
# Pull evaluation containers (no local installation needed)
docker pull nvcr.io/nvidia/eval-factory/simple-evals:${DOCKER_TAG}
docker pull nvcr.io/nvidia/eval-factory/lm-evaluation-harness:${DOCKER_TAG}
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:${DOCKER_TAG}
# [snippet-end]

# Verify containers are pulled
echo "Verifying container images..."
docker images | grep "eval-factory" && echo "✓ Containers pulled successfully"

