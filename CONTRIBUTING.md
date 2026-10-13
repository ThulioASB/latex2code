# Contributing

Thanks for helping improve latex2code. Bug reports with a minimal LaTeX input, the backend you
used, and the output you expected are the most useful contributions.

## Development setup

```bash
git clone https://github.com/ThulioASB/latex2code.git
cd latex2code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

Use the `dev` extra rather than installing pytest by hand: `pyproject.toml` configures pytest to
run with `--cov`, which needs `pytest-cov`. The PyTorch and JAX tests skip themselves when those
frameworks are missing; install them to run those tests:

```bash
pip install -e ".[torch,jax]"
```

## Running checks

```bash
pytest
black --check src tests
ruff check src tests
mypy src
```

## Guidelines

- Add a test with every behavior change. Transpilation tests should `exec` the generated source
  and check values, not only match strings.
- If you add a new function or LaTeX construct, add a printer for every backend, or make the
  unsupported backends fail with `CodeGenerationError` rather than emitting wrong code.
- Update `docs/` and `CHANGELOG.md` for user-visible changes.
- Keep the scope focused: this is an expression transpiler, not a LaTeX document parser or a
  computer algebra system.
