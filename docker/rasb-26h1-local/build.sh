#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RASB_ROOT="${1:-${DATASET_DIR:-${RASB_ROOT:-}}}"
IMAGE_TAG="${IMAGE_TAG:-localhost/rasb-26h1:local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

if [ -z "${RASB_ROOT}" ]; then
  echo "RASB root is required." >&2
  echo "Pass it as the first argument or set DATASET_DIR." >&2
  exit 1
fi

if [ ! -f "${RASB_ROOT}/benchmark.py" ]; then
  echo "RASB root is invalid: ${RASB_ROOT}" >&2
  echo "Expected benchmark.py under the RASB root." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}/packages" "${BUILD_DIR}/rasb-26h1"
cp -R "${REPO_ROOT}/packages/nemo-evaluator" "${BUILD_DIR}/packages/nemo-evaluator"

tar \
  --exclude='.env' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.tmp' \
  --exclude='results' \
  -C "${RASB_ROOT}" \
  -cf - . | tar -C "${BUILD_DIR}/rasb-26h1" -xf -

cp "${SCRIPT_DIR}/Dockerfile" "${BUILD_DIR}/Dockerfile"

docker build -t "${IMAGE_TAG}" "${BUILD_DIR}"
echo "Built ${IMAGE_TAG}"
