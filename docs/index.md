# NeMo Evaluator

Welcome to NeMo Evaluator - NVIDIA's comprehensive platform for AI model evaluation and benchmarking.

## Overview

NeMo Evaluator is NVIDIA's open-source evaluation stack that provides a unified platform for AI model evaluation and benchmarking. It consists of two core libraries: nemo-evaluator (the core evaluation engine) and nemo-evaluator-launcher (the CLI and orchestration layer). Together, these components enable consistent, scalable, and reproducible evaluation of GenAI models spanning LLMs, VLMs, agentic AI, and retrieval systems.

### Quickstart

With a single command, you can select and run any of the 100+ supported benchmarks across 18 open‑source harnesses, executed in fully transparent, open‑source Docker containers to guarantee reproducible results. See the launcher documentation: [nemo-evaluator-launcher](nemo-evaluator-launcher/index.md).

If you want to evaluate a model on a specific benchmark without depolying the model yourself, you can select an OpenAI‑compatible endpoint from the extensive catalog on [NVIDIA Build](https://build.nvidia.com/models). These ready‑to‑use endpoints integrate seamlessly with NeMo Evaluator, enabling you to set up end‑to‑end evaluations in a matter of minutes. See the [Quickstart](nemo-evaluator-launcher/quickstart.md).


### NeMo Evaluator at a glance

- **Purpose**: Fully transparent and reproducible experiment setups for LLM evaluation

- **Key pillars**:
  - Open‑source Docker containers for all benchmarks
  - Reproducible configurations saved per run
  - Transparent evaluation process
  - Consistent containerized environments

- **Architecture highlights**:
  - Model is separate from the evaluation container; communication via OpenAI‑compatible API
  - Execution environments: Local, Slurm, Lepton, or your custom execution backend

## System Architecture

NeMo Evaluator consists of two main libraries:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NeMo Evaluator Ecosystem                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                    ┌─────────────────┐     │
│  │   nemo-         │                    │   nemo-         │     │
│  │  evaluator      │                    │  evaluator-     │     │
│  │                 │                    │  launcher       │     │
│  │  Core evaluation│◄──────────────────►│  User interface │     │
│  │  engine &       │                    │  & orchestration│     │
│  │  adapters       │                    │                 │     │
│  └─────────────────┘                    └─────────────────┘     │
│           │                              │                      │
│           ▼                              ▼                      │
│  ┌─────────────────┐                    ┌─────────────────┐     │
│  │   Evaluation    │                    │   CLI & API     │     │
│  │   Frameworks    │                    │   Interface     │     │
│  │   & Containers  │                    │                 │     │
│  └─────────────────┘                    └─────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```


