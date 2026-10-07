# latex2code 🧮 ➡️ 🐍

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight Python library and CLI tool that transpiles a supported subset of LaTeX mathematical expressions into Python functions.

Designed for scientific computing, machine learning researchers, and engineers who want to bridge the gap between academic papers and executable code.

---

## Features

- 📐 **Common Math:** Translates arithmetic, fractions, powers, roots, and common functions such as `\sin`, `\cos`, `\tan`, `\exp`, and `\log`.
- 🔢 **Greek Symbols:** Supports common Greek-letter commands as symbols in expressions.
- 🔁 **Matrix Support:** Converts LaTeX matrix environments (`\begin{pmatrix}`, `\begin{matrix}`) into `numpy.ndarray` objects.
- 🧮 **Symbolic Operations:** Uses SymPy to evaluate supported derivatives, integrals, and finite sums symbolically.
- 🏷️ **Type Annotations:** Automatically generates functions with PEP 484 type hints.
- 💻 **Command Line Interface:** Transpiles formulas directly from your terminal.

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

---

## CLI Usage

You can run `latex2code` directly from your command line:

```powershell
latex2code "\frac{a}{b}" --name divide

```

### Options

| Flag | Long Option | Description |
| --- | --- | --- |
| `-n` | `--name` | Name of the generated Python function (default: `formula`). |
|  | `--no-types` | Disable type hints in the generated signature. |
| `-p` | `--numpy` | Force generation with NumPy functions/arrays. |

---

## Running Tests

Run the test suite using `pytest`:

```powershell
pytest
```

## Supported Scope and Limitations

- Input is parsed by SymPy's LaTeX parser. Support is limited to mathematical expressions it can parse and the commands recognized by this package; document markup, arbitrary custom functions, and many advanced LaTeX constructs are outside the scope.
- Greek symbols may become function parameters rather than built-in constants. Inspect the generated signature and provide the intended values.
- Derivatives and integrals are evaluated symbolically when SymPy can do so. An indefinite integral returns one antiderivative and does not add the arbitrary constant of integration.
- Finite sums are emitted as executable Python sum expressions. Matrices are emitted as NumPy arrays.
- Generated Python is not a mathematical proof or a guarantee of numerical stability. Review and test output before relying on it in research or production calculations.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
