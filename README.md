# latex2code 🧮 ➡️ 🐍

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight Python library and CLI tool to transpile LaTeX mathematical expressions into executable, type-annotated Python functions.

Designed for scientific computing, machine learning researchers, and engineers who want to bridge the gap between academic papers and executable code.

---

## Features

- 📐 **Trigonometry & Calculus:** Translates fractions, powers, roots, trigonometric functions (`\sin`, `\cos`, `\tan`), and exponents into Python `math` module calls.
- 🔢 **Matrix Support:** Converts LaTeX matrix environments (`\begin{pmatrix}`, `\begin{matrix}`) into `numpy.ndarray` objects.
- 🔁 **Summations:** Transpiles `\sum_{i=a}^{b}` into executable iterative loops.
- 🏷️ **Type Annotations:** Automatically generates functions with PEP 484 type hints.
- 💻 **Command Line Interface:** Transpiles formulas directly from your terminal.

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

### Matrix Transpilation

```python
from latex2code import transpile_latex

latex_matrix = r"\begin{pmatrix} x^2 & 1 \\ 0 & y \end{pmatrix}"
code = transpile_latex(latex_matrix, function_name="create_matrix")
print(code)

```

**Output:**

```python
import numpy as np

def create_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
