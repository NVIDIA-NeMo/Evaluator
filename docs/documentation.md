#  Build and Test the Documentation

This guide provides instructions for contributing to the NeMo Eval documentation. It covers the essential steps for building the documentation locally, serving it with a live-reloading server for easier editing, and running tests embedded within Python docstrings to ensure all code examples are accurate.

## Build the Documentation

The following sections describe how to set up and build the NeMo Eval documentation.

Switch to the documentation source folder and generate the HTML output.

```sh
cd docs/
uv run --project ../packages/nemo-evaluator --group docs sphinx-build . _build/html
```

* The resulting HTML files are generated in a `_build/html` folder that is created under the project `docs/` folder.
* The generated Python API docs are placed in `apidocs` under the `docs/` folder.

## Serve the Documentation with Live Reload

When writing documentation, it can be helpful to serve the documentation and have it update live while you edit.

To do so, run:

```sh
cd docs/
uv run --project ../packages/nemo-evaluator --group docs sphinx-autobuild . _build/html --port 12345 --host 0.0.0.0
```

Open a web browser and go to `http://${HOST_WHERE_SPHINX_COMMAND_RUN}:12345` to view the output.


## Run Tests in Python Docstrings

We also run tests in our Python docstrings. You can run them with:

```sh
cd docs/
uv run --project ../packages/nemo-evaluator --group docs sphinx-build -b doctest . _build/doctest
```

## Write Tests in Python Docstrings

Any code in triple backtick blocks with the `{doctest}` directive will be tested. The format follows Python's doctest module syntax, where `>>>` indicates Python input and the following line shows the expected output. Here's an example:

```python
def add(x: int, y: int) -> int:
    """
    Adds two integers together.

    Args:
        x (int): The first integer to add.
        y (int): The second integer to add.

    Returns:
        int: The sum of x and y.

    Examples:
    ```{doctest}
    >>> from nemo_eval.made_up_package import add
    >>> add(1, 2)
    3
    ```

    """
    return x + y
```
