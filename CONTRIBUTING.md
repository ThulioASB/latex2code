# Contributing to latex2code

Thank you for considering contributing to `latex2code`! We welcome bug fixes, new LaTeX operator mappings, backend improvements, and documentation enhancements.

---

## Development Setup

1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/ThulioASB/latex2code.git
   cd latex2code
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,torch,jax]"
2. Make your changes in `src/latex2code/`. If you are adding a new LaTeX function or operator:

- Register the SymPy expression node in `src/latex2code/parsing.py`.

- Add backend mappings in `src/latex2code/backends.py`.

- Define printer translation rules in `src/latex2code/printers.py`.

- Add unit tests in `tests/test_core.py`.

## Code Quality Checks
Before submitting a Pull Request, ensure all linters, type checks, and tests pass locally:

```bash
# 1. Format code automatically
python -m black src tests

# 2. Check for lint errors (and auto-fix imports if needed)
python -m ruff check --fix src tests

# 3. Type check with mypy
python -m mypy src

# 4. Run the full test suite
python -m pytest
```
## Commit Guidelines
We recommend using clear, conventional commit messages:

- `feat: add support for \operatorname{...}`

- `fix: resolve issue with piecewise evaluation`

- `docs: update README installation steps`

- `style: format imports and code with black/ruff`

## Pull Request Process
1. Fork the repo and create your branch from `main`.

2. Ensure all 4 check commands above pass cleanly.

3. Open a PR with a description of your changes and reference any relevant issue.


---