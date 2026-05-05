# Context Map

## Contexts

- [Common Evaluation Context](./CONTEXT.md) - shared domain language for NeMo Evaluator across benchmarks, solvers, services, sandboxes, scoring, reporting, and orchestration.
- [Adapter Context](./src/nemo_evaluator/adapters/CONTEXT.md) - adapter proxy and interceptor vocabulary for service traffic middleware.
- [Environment Context](./src/nemo_evaluator/environments/CONTEXT.md) - environment adapters and Bring Your Own Benchmark vocabulary.

## Relationships

- **Common Evaluation Context -> implementation contexts**: per-module context files may extend this shared vocabulary, but should not redefine common terms without calling out the conflict.
- **Adapter Context -> Common Evaluation Context**: adapter-specific terms extend **Service** and **Solver** traffic vocabulary without redefining service ownership.
- **Environment Context -> Common Evaluation Context**: environment-specific terms extend **Benchmark**, **Environment**, **Seed**, **Verification**, **Reward**, and **Scoring**.
