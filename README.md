# latex2code 🧮 ➡️ 🐍

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight Python library and CLI tool that transpiles a supported subset of LaTeX mathematical expressions into Python functions.

Designed for scientific computing, machine learning researchers, and engineers who want to bridge the gap between academic papers and executable code.

---

## Features

- 📐 **Common Math:** Translates arithmetic, fractions, powers, roots, limits, binomial coefficients, and common functions such as `\sin`, `\cos`, `\tan`, `\exp`, `\log`, `\operatorname{erf}`, `\max`, and hyperbolic functions.
- 🔢 **Greek Symbols:** Supports common Greek-letter commands as symbols in expressions.
- 🔁 **Matrix Support:** Converts LaTeX matrix environments (`\begin{pmatrix}`, `\begin{bmatrix}`, `\begin{matrix}`) into `numpy.ndarray` objects.
- 🧩 **Piecewise Expressions:** Recognizes `\begin{cases} ... \end{cases}` and emits Python conditional expressions.
- 🔒 **Absolute Values & Floor/Ceiling:** Handles common wrappers like `\left|x\right|`, `\left\lfloor x \right\rfloor`, and `\left\lceil x \right\rceil`.
- 🧮 **Symbolic Operations:** Uses SymPy to evaluate supported derivatives, limits, integrals, and finite products; finite sums are emitted as executable Python.
- 🏷️ **Type Annotations:** Automatically generates functions with PEP 484 type hints.
- 💻 **Command Line Interface:** Transpiles formulas directly from your terminal and accepts input from stdin.

This is a focused expression transpiler, not a complete LaTeX implementation or a computer algebra system.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ThulioASB/latex2code.git
cd latex2code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

```

---

## Usage

### As a Python Library

```python
from latex2code import transpile_latex

# Transpile a fraction with trigonometric functions
code = transpile_latex(r"\frac{\sin(x) + \cos(y)}{\sqrt{z}}", function_name="calculate_wave")
print(code)

```

**Output:**

```python
import math

def calculate_wave(x: float, y: float, z: float) -> float:
    return (math.sin(x) + math.cos(y))/math.sqrt(z)

```

### Greek Symbols

```python
from latex2code import transpile_latex

code = transpile_latex(r"\sin(\pi x) + \theta", function_name="signal_response")
print(code)
```

Greek-letter commands can become function arguments. For example, `\pi` is not guaranteed to be treated as the built-in numeric constant; pass the desired value (such as `math.pi`) as an argument when it appears as a symbol.

### Matrix Transpilation

```python
from latex2code import transpile_latex

latex_matrix = r"\begin{pmatrix} x^2 & 1 \\ 0 & y \end{pmatrix}"
code = transpile_latex(latex_matrix, function_name="create_matrix", type_hints=False)
print(code)

```

**Output:**

```python
import numpy as np

def create_matrix(x, y):
    return np.array([[x**2, 1], [0, y]])

```

### Piecewise Expressions

```python
from latex2code import transpile_latex

code = transpile_latex(
    r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}",
    function_name="abs_value",
    type_hints=False,
)
print(code)
```

**Output:**

```python
def abs_value(x):
    return (x) if (x > 0) else (-x)
```

### Limits and Binomial Coefficients

```python
from latex2code import transpile_latex

limit_code = transpile_latex(r"\lim_{x \to 0} \frac{\sin(x)}{x}", function_name="sinc_limit", type_hints=False)
print(limit_code)

binomial_code = transpile_latex(r"\binom{n}{k}", function_name="choose", type_hints=False)
print(binomial_code)

product_code = transpile_latex(r"\prod_{i=1}^{n} i", function_name="factorial_product", type_hints=False)
print(product_code)
```

**Output:**

```python
def sinc_limit():
    return 1

import math

def choose(n, k):
    return (math.gamma(n + 1)/(math.gamma(k + 1)*math.gamma(-k + n + 1)))

import math

def factorial_product(n):
    return math.factorial(n)
```

---

## CLI Usage

You can run `latex2code` directly from your command line:

```powershell
latex2code "\frac{a}{b}" --name divide
```

You can also pipe input from standard input:

```powershell
"\sin(x) + \cos(y)" | latex2code - --name signal
```

Write the generated source directly to a Python file with `--output`:

```powershell
latex2code "\frac{a}{b}" --name divide --output divide.py
```

When `--numpy` is enabled, generated annotations accept either scalar floats or NumPy arrays. Matrix functions always annotate their result as `numpy.ndarray`; their entries accept arrays too when NumPy mode is enabled.

### Options

| Flag | Long Option | Description |
| --- | --- | --- |
| `-n` | `--name` | Name of the generated Python function (default: `formula`). |
|  | `--no-types` | Disable type hints in the generated signature. |
| `-p` | `--numpy` | Force generation with NumPy functions/arrays. |
| `-o` | `--output` | Write generated Python source to a file instead of stdout. |

---

## Running Tests

Run the test suite using `pytest`:

```powershell
pytest
```

## Supported Scope and Limitations

- Input is parsed by SymPy's LaTeX parser. Support is limited to mathematical expressions it can parse and the commands recognized by this package; document markup, arbitrary custom functions, and many advanced LaTeX constructs are outside the scope.
- Unsupported commands and custom functions raise `UnsupportedLaTeXFeatureError`, an `InvalidLaTeXSyntaxError` subclass, so callers can report unsupported syntax separately from malformed input.
- Piecewise cases without an explicit default remain undefined outside their listed conditions; generated Python returns `None` there, while NumPy mode represents those values as `numpy.nan`.
- Greek symbols may become function parameters rather than built-in constants. Inspect the generated signature and provide the intended values.
- Derivatives and integrals are evaluated symbolically when SymPy can do so. An indefinite integral returns one antiderivative and does not add the arbitrary constant of integration.
- Finite sums are emitted as executable Python sum expressions. Matrices are emitted as NumPy arrays.
- Finite products are symbolically evaluated when SymPy can reduce them; unresolved symbolic products raise `CodeGenerationError` rather than returning incomplete Python.
- NumPy mode emits array-aware operations for elementwise expressions; special functions unavailable in NumPy directly use vectorized standard-library implementations. Finite sums still use scalar iteration bounds.
- Generated Python is not a mathematical proof or a guarantee of numerical stability. Review and test output before relying on it in research or production calculations.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
