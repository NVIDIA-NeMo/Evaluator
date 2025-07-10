# Contributing To NeMo Eval

Thanks for your interest in contributing to NeMo Eval!

## 🛠️ Setting Up Your Environment

### Local workstation

NeMo Eval uses [uv](https://docs.astral.sh/uv/) for package management.

You can configure uv with the following commands:

```bash
uv sync --only-group build  # Installs build dependencies required by TransformerEngine
uv sync
```

On a machine with CUDA, you can additionally sync TransformerEngine:

```bash
uv sync --extra te
```

### Alternative: Development Container

For containerized development, use our Dockerfile for building your own container:

```bash
docker build \
    -f docker/Dockerfile.ci \
    -t nemo-eval \
    --build-arg INFERENCE_FRAMEWORK=inframework \
    .
```

Start your container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nemo-eval
```

## 📦 Dependencies management

We use [uv](https://docs.astral.sh/uv/) for managing dependencies. For reproducible builds, our project tracks the generated `uv.lock` file in the repository.  
On a weekly basis, the CI attemps an update of the lock file to test against upstream dependencies.

New required dependencies can be added by `uv add $DEPENDENCY`.

New optional dependencies can be added by `uv add --optional --extra $EXTRA $DEPENDENCY`.

`EXTRA` refers to the subgroup of extra-dependencies to which you're adding the new dependency.
Example: For adding a TE specific dependency, run `uv add --optional --extra te $DEPENDENCY`.

Alternatively, the `pyproject.toml` file can also be modified directly.

Adding a new dependency will update UV's lock-file. Please check this into your branch:

```bash
git add uv.lock pyproject.toml
git commit -m "build: Adding dependencies"
git push
```

### 🧹 Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. CI does not auto-fix linting and formatting issues, but most issues can be fixed by running the following command:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Note: If `ruff` is missing, please follow the [installation](#local-workstation) guide.

### 📝 Documentation

**Important**: All new key features (ex: enabling a new inference optimized library, enabling a new deployment option) must include documentation update (either a new doc or updating an existing one). This document update should:

- Explain the motivation and purpose of the feature
- Outline the technical approach and architecture
- Provide clear usage examples and instructions for users
- Document internal implementation details where appropriate

This ensures that all significant changes are well-thought-out and properly documented for future reference. Comprehensive documentation serves two critical purposes:

1. **User Adoption**: Helps users understand how to effectively use the library's features in their projects
2. **Developer Extensibility**: Enables developers to understand the internal architecture and implementation details, making it easier to modify, extend, or adapt the code for their specific use cases

Quality documentation is essential for both the usability of NeMo Eval and its ability to be customized by the community.

### Local development

Make sure to have [uv](https://docs.astral.sh/uv/) installed. You can build and inspect the documentation locally with the following commands:

```bash
cd docs/
uv run --only-group docs sphinx-autobuild . _build/html
```

## Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

## Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -e ".[dev,test]"
   ```
4. Run pre-commit hooks:
   ```bash
   pre-commit install
   ```
5. Make your changes and add tests
6. Submit a pull request

## Testing

### Running Tests

```bash
# Run all tests
pytest tests

# Run unit tests only
pytest tests/unit_tests/

# Run functional tests only
pytest tests/functional_tests/

# Run with coverage
pytest --cov=nemo_eval tests
```

### Test Scripts

```bash
# Unit tests on CPU
bash tests/unit_tests/L0_Unit_Tests_CPU.sh

# Unit tests on GPU
bash tests/unit_tests/L0_Unit_Tests_GPU.sh

# Functional tests on GPU
bash tests/functional_tests/L2_Functional_Tests_GPU.sh
```

### Testing Guidelines

- Write unit tests for new functionality
- Ensure all tests pass before submitting
- Add integration tests for complex features
- Follow existing test patterns

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit** for automated checks
