# latex2code

[![PyPI version](https://badge.fury.io/py/latex2code.svg)](https://badge.fury.io/py/latex2code)
[![Tests Status](https://github.com/ThulioASB/latex2code/actions/workflows/tests.yml/badge.svg)](https://github.com/ThulioASB/latex2code/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight Python library and CLI that transpiles a supported subset of LaTeX math into
readable, standalone Python functions, for plain Python, NumPy, PyTorch, or JAX.

Aimed at researchers and engineers who want to get from a formula in a paper to executable code
they can inspect, test, and commit.

```python
from latex2code import transpile_latex

print(transpile_latex(r"\frac{\sin(x) + \cos(y)}{\sqrt{z}}", function_name="calculate_wave"))
```

```python
import math

def calculate_wave(x: float, y: float, z: float) -> float:
    return (math.sin(x) + math.cos(y))/math.sqrt(z)
```

This is a focused expression transpiler, not a complete LaTeX implementation or a computer
algebra system.

## Features

- **Math:** arithmetic, fractions, powers, roots (including nth roots), trigonometric, inverse,
  reciprocal and hyperbolic functions, `exp`/`log`, `erf`, `gamma`, `beta`, `max`/`min`, absolute
  values, floor/ceiling, binomial coefficients.
- **Calculus and series:** derivatives, limits, integrals and finite products are evaluated
  symbolically with SymPy; finite sums are emitted as executable Python.
- **Structures:** matrix and array environments become `ndarray` objects; `cases` becomes a
  piecewise expression.
- **Symbols:** Greek letters become variables; `\pi` and `e` are numeric constants. Rename
  arguments with `variable_map`, or fix their order with `args`.
- **ML operators:** `sigmoid`, `relu`, `softplus`, `softmax`, `gelu`, `silu`, `logit`,
  `softsign`, `swish` via `\operatorname{...}`.
- **Four backends:** `python` (default), `numpy`, `torch`, `jax`. Torch and JAX output uses
  framework primitives so autodiff can trace it.
- **Inspection and diagnostics:** see the parsed expression tree, variables, and warnings
  before trusting the output. Errors carry a line, column, source excerpt, and
  "did you mean" suggestions.
- **Generated-code niceties:** PEP 484 type hints, and an optional docstring that records the
  source LaTeX (`docstring=True` / `--docstring`).

## Installation

```bash
pip install latex2code
```

Optional framework backends:

```bash
pip install "latex2code[torch]"
pip install "latex2code[jax]"
```

Requirements: Python 3.10+, SymPy, NumPy, and `antlr4-python3-runtime` 4.11.x (needed by SymPy's
LaTeX parser; installed automatically). The generated code imports only what its backend needs:
the standard library for `python` (plus NumPy if the formula is a matrix), NumPy for `numpy`,
PyTorch for `torch`, and JAX for `jax`.

From source:

```bash
git clone https://github.com/ThulioASB/latex2code.git
cd latex2code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Quick start

### Library

```python
import numpy as np
from latex2code import compile_latex, inspect_latex, transpile_latex

# Source code as a string
source = transpile_latex(r"\operatorname{sigmoid}(x)", backend="numpy")

# A callable that also keeps its source and metadata
formula = compile_latex(r"\frac{\sin(x)}{\sqrt{x^2 + 1}}", backend="numpy")
result = formula(np.linspace(0, 10, 100))
print(formula.source)

# Check how an expression was interpreted
details = inspect_latex(r"\sin(\pi x) + \theta")
print(details.parsed_expression, details.variables, details.warnings)
```

Binomials stay exact for integers and fall back to the gamma function for real arguments:

```python
print(transpile_latex(r"\binom{n}{k}", function_name="choose", type_hints=False))
```

```python
import math

def choose(n, k):
    return (math.comb(n, k) if isinstance(n, int) and isinstance(k, int) and n >= 0 and k >= 0 else math.gamma(n + 1)/(math.gamma(k + 1)*math.gamma(-k + n + 1)))
```

More examples (constants, matrices, piecewise, activations, training loops, a Gaussian
density) are in [docs/usage.md](docs/usage.md).

### Command line

```bash
latex2code '\frac{a}{b}' --name divide
echo '\sin(x) + \cos(y)' | latex2code - --name signal
latex2code '\sin(x)^2 + x' --backend torch --output differentiable.py
latex2code '\theta + x' --map-variable theta=angle
latex2code '\frac{a}{b}' --docstring
latex2code '\sin(\pi x) + \theta' --inspect
latex2code --interactive
```

| Flag | Long option | Description |
| --- | --- | --- |
| `-n` | `--name` | Name of the generated function (default: `formula`). |
|  | `--no-types` | Omit type hints. |
|  | `--docstring` | Record the source LaTeX in the function's docstring. |
|  | `--backend` | `python`, `numpy`, `torch`, or `jax` (default: `python`). |
| `-o` | `--output` | Write the source to a file instead of stdout. |
|  | `--inspect` | Print a JSON report (parsed expression, variables, warnings, source). |
|  | `--map-variable SYMBOL=NAME` | Rename a parsed variable; repeatable. |
| `-i` | `--interactive` | Read expressions until `quit`, `exit`, or end-of-file. |

## Why not just SymPy?

SymPy's `parse_latex` plus `lambdify` already covers simple formulas, and if that is all you
need, use it. latex2code is for when you also want:

- **Source you can read and commit.** The output is a plain `def` with type hints and,
  optionally, the original LaTeX in its docstring, not an opaque closure.
- **Framework targets.** NumPy, PyTorch, and JAX output built from native primitives, with
  stable implementations of `sigmoid`, `softplus`, and `swish`.
- **Visibility into the parse.** `inspect_latex` / `--inspect` show the expression tree,
  argument order, and warnings, so a silently misread formula does not become silent wrong code.
- **Strict parsing and useful errors.** Trailing junk is rejected rather than dropped, and
  failures point at a line and column.
- **Control over the signature.** `variable_map`, `args`, and `constants` decide what the
  function takes without changing the mathematics.

## Security

`compile_latex` executes the Python it generates. The generator validates every identifier and
builds code from a parsed expression tree, and the optional docstring is escaped so input
cannot break out of it, but this is not a sandbox. As with any code generator, do not compile
or `exec` output derived from LaTeX you do not trust.

## Documentation

- [docs/usage.md](docs/usage.md): detailed usage and worked examples
- [docs/backends.md](docs/backends.md): what each backend supports and where semantics differ
- [docs/limitations.md](docs/limitations.md): supported scope and known limitations
- [examples/basic_usage.py](examples/basic_usage.py): a runnable tour
- [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Distributed under the MIT License. See `LICENSE` for more information.
