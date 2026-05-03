# Context Map

## Contexts

- [Common Evaluation Context](./CONTEXT.md) - shared domain language for NeMo Evaluator across benchmarks, solvers, services, sandboxes, scoring, reporting, and orchestration.
- [Adapter Context](./src/nemo_evaluator/adapters/CONTEXT.md) - adapter proxy and interceptor vocabulary for service traffic middleware.
- [Built-in Benchmark Context](./src/nemo_evaluator/benchmarks/CONTEXT.md) - first-party benchmark vocabulary, including benchmark-specific task, workspace, and grading concepts.
- [Environment Context](./src/nemo_evaluator/environments/CONTEXT.md) - environment adapters and Bring Your Own Benchmark vocabulary.

## Relationships

- **Common Evaluation Context -> implementation contexts**: per-module context files may extend this shared vocabulary, but should not redefine common terms without calling out the conflict.
- **Adapter Context -> Common Evaluation Context**: adapter-specific terms extend **Service** and **Solver** traffic vocabulary without redefining service ownership.
- **Built-in Benchmark Context -> Common Evaluation Context**: first-party benchmark terms extend **Benchmark**, **Problem**, **Seed**, **Verification**, **Reward**, and **Scoring**.
- **Built-in Benchmark Context -> Environment Context**: built-in benchmarks may use the BYOB registration machinery, but benchmark-specific task language belongs in the built-in benchmark context.
- **Environment Context -> Common Evaluation Context**: environment-specific terms extend **Benchmark**, **Environment**, **Seed**, **Verification**, **Reward**, and **Scoring**.
