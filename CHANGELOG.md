# Changelog

## 0.1.0 (unreleased)

Initial release.

- `EvalEnvironment` abstraction with `seed()`/`verify()` protocol
- Built-in benchmarks: GSM8K, TriviaQA
- Async evaluation runner with concurrent model calls
- `ModelClient` with connection pooling, retry logic, backoff
- Full observability: trajectories, runtime stats, failure analysis
- Scoring primitives: `math_equal`, `exact_match`, `extract_answer`
- Metrics: `pass@k`, bootstrap confidence intervals, category breakdown
- Adapters: Gym, PI (verifiers), NeMo Skills, legacy containers
- CLI: `nel run`, `nel serve`, `nel validate`, `nel regression`, `nel slurm`
- Distributed evaluation via SLURM, Kubernetes, Ray
- Regression comparison with CI overlap detection
- Sphinx documentation with tutorials and deployment guides
