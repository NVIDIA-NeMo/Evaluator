# Context Map

## Contexts

- [Common Evaluation Context](./CONTEXT.md) - shared domain language for NeMo Evaluator across benchmarks, solvers, services, sandboxes, scoring, reporting, and orchestration.

## Relationships

- **Common Evaluation Context -> implementation contexts**: per-module context files may extend this shared vocabulary, but should not redefine common terms without calling out the conflict.
