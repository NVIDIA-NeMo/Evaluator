# Contributing to nv-eval

NV-Eval uses:
* [Poetry](https://python-poetry.org/) for packaging and dependency management
* [black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/) for code formatting

## Setup

1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install depdendencies
```bash
poetry install
```

3. Activate virtualenv
```bash
eval $(poetry env activate)
```

## Code Formatting

### Format manually
```bash
poetry run black .
poetry run isort .
```


### Pre-commit hooks
1. Install pre-commit
```bash
poetry run pre-commit install
```

2. Run manually
```
poetry run pre-commit run --all-files
```
