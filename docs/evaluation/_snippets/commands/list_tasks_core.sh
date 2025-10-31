#!/bin/bash
# Task discovery commands for NeMo Evaluator
# FIXME(martas): Hard-code the container version

# [snippet-start]
# List benchmarks available in the container
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:25.10 nemo-evaluator ls
# [snippet-end]
