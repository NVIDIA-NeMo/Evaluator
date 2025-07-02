#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

source ${UV_PROJECT_ENVIRONMENT}/bin/activate

# Nemo-run has conflicting dependencies to export-deploy:
# They collide on nvidia-pytriton (export-deploy) and torchx (nemo-run)
# via urllib3.
uv pip install nemo-run
