# latex2code 🧮 ➡️ 🐍

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A lightweight Python library and CLI tool that transpiles a supported subset of LaTeX mathematical expressions into Python functions.

Designed for scientific computing, machine learning researchers, and engineers who want to bridge the gap between academic papers and executable code.

---

## Features

- 📐 **Common Math:** Translates arithmetic, fractions, powers, roots, limits, binomial coefficients, and common functions such as `\sin`, `\cos`, `\tan`, `\exp`, `\log`, `\operatorname{erf}`, `\max`, and hyperbolic functions.
- 🔢 **Greek Symbols & Constants:** Common Greek-letter commands become variables; `\pi` and `e` are numeric constants.
- 🔁 **Matrix & Array Support:** Converts LaTeX matrix environments (`\begin{pmatrix}`, `\begin{bmatrix}`, `\begin{matrix}`) and tabular/array layouts like `\begin{array}{cc} ... \end{array}` into `numpy.ndarray` objects.
- 🧩 **Piecewise Expressions:** Recognizes `\begin{cases} ... \end{cases}` and emits Python conditional expressions.
- 🔒 **Absolute Values & Floor/Ceiling:** Handles common wrappers like `\left|x\right|`, `\left\lfloor x \right\rfloor`, and `\left\lceil x \right\rceil`.
- 🧮 **Symbolic Operations:** Uses SymPy to evaluate supported derivatives, limits, integrals, finite products, and nth roots; finite sums are emitted as executable Python.
- 📐 **Extended Trig & Inverse Functions:** Supports reciprocal and inverse trig functions such as `\csc`, `\arccot`, `\arcsec`, and `\arccsc`.
- 🤖 **Machine-Learning Functions:** Supports `\operatorname{sigmoid}(x)`, `\operatorname{relu}(x)`, `\operatorname{softplus}(x)`, `\operatorname{logit}(x)`, `\operatorname{softsign}(x)`, and `\operatorname{swish}(x)` in scalar, NumPy, PyTorch, and JAX modes.
- 🏷️ **Type Annotations:** Automatically generates functions with PEP 484 type hints.
- 🔍 **Inspection Reports:** Shows the parsed symbolic expression, variables, warnings, and generated Python source.
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

Install an optional differentiable backend with one of the extras:

```powershell
pip install -e ".[torch]"
pip install -e ".[jax]"
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

For direct evaluation in a notebook or application, use `compile_latex`. The returned object is
callable and retains the generated source and inspection details:

```python
from latex2code import compile_latex

formula = compile_latex(r"\sin(x) + x^2")
print(formula(0.5))
print(formula.info.variables)
print(formula.source)
```

Choose a backend just as with `transpile_latex`; optional frameworks must be installed when
compiling a formula for that backend. The callable uses only the arguments supplied by the caller
and does not automatically move or cast array/tensor inputs.

### Constants and Greek Variables

```python
from latex2code import transpile_latex

code = transpile_latex(r"\sin(\pi x) + e + \theta", function_name="signal_response")
print(code)
```

Standard constants `\pi` and `e` are emitted as numeric constants and are not function arguments. Other Greek-letter symbols such as `\theta` remain variables and appear in the generated function signature.

### Inspecting a Transpilation

Use the inspection API to check how an expression was interpreted before using its generated source:

```python
from latex2code import inspect_latex

details = inspect_latex(r"\sin(\pi x) + \theta")
print(details.input_expression)
print(details.backend)
print(details.parsed_expression)
print(details.expression_tree)
print(details.variables)
print(details.generated_code)
print(details.warnings)
```

The CLI's `--inspect` option prints the same information as JSON. Parse and unsupported-feature
errors include a line and column with a short source excerpt when the location can be identified:

```powershell
latex2code "\sin(\pi x) + \theta" --inspect
```

Use `variable_map` to give parsed symbols clearer Python argument names without changing the
mathematics. Keys are parsed symbol names, and values must be unique Python identifiers:

```python
from latex2code import compile_latex

model = compile_latex(
    r"\theta + \lambda",
    variable_map={"theta": "angle", "lambda": "wavelength"},
)
print(model(angle=0.5, wavelength=2.0))
```

Inspection reports include `expression_tree`, a JSON-compatible tree of SymPy operations and
arguments, as well as the readable `parsed_expression` string.

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

### Machine-Learning Activations

Common activation and link functions can be generated for scalar, NumPy, PyTorch, or JAX inputs:

```python
from latex2code import transpile_latex

code = transpile_latex(
    r"\operatorname{sigmoid}(x)",
    function_name="logistic_probability",
    use_numpy=True,
)
print(code)
```

The default `python` backend emits ordinary Python math. `backend="numpy"` supports NumPy arrays.
The optional `torch` and `jax` backends generate operations using framework primitives so tensor
autodiff can trace expressions; pass tensors/arrays of the chosen framework as inputs. These
backends do not imply that every supported LaTeX construct has equivalent batching or gradient
semantics. Named `sigmoid`, `softplus`, and `swish` operators use stable scalar or
framework-native implementations. Writing equivalent expanded formulas manually (such as
`log(1 + exp(x))`) is not guaranteed to receive this stabilization. Check input domains for
`logit`, logarithms, and roots; the transpiler does not silently clip inputs.

```python
torch_code = transpile_latex(r"\sin(x)^2 + x", backend="torch")
jax_code = transpile_latex(r"\sin(x)^2 + x", backend="jax")
```

### Using Generated Code in Training or Analysis

Choose tensor dtype/device and array shape in your application; generated functions preserve
framework operations and do not move or silently cast tensor inputs:

```python
import torch

source = transpile_latex(r"\operatorname{softplus}(x) + x^2", backend="torch")
namespace = {}
exec(source, namespace)
formula = namespace["formula"]

x = torch.tensor([0.25, 1.0], dtype=torch.float64, device="cpu", requires_grad=True)
loss = formula(x).mean()
loss.backward()
```

For JAX, generated functions can be passed to `jax.jit`, `jax.vmap`, and `jax.grad` when the
expression uses supported JAX primitives. Test the actual shapes and dtypes used by your workload.

Piecewise functions use eager elementwise selection (`where`) for array/tensor inputs: both branch
expressions may be evaluated before selection. Avoid invalid operations in inactive branches (such
as taking `log(x)` in a branch where `x <= 0`); use a formula with valid branch expressions or
write a framework-specific implementation when true lazy control flow is required. Transpiling
piecewise expressions for NumPy, PyTorch, or JAX emits a `PiecewiseEvaluationWarning`. Automatic
domain clipping is intentionally not performed because it changes the formula and its gradients.

### Reproducible Scientific Example: Gaussian Density

This normal probability density uses `x`, `mu`, and `sigma` in the same units; its output has
inverse units. Require `sigma > 0`. At the mean, with `sigma = 2`, the density is
`1 / (2 sqrt(2 pi))`, approximately `0.19947114020071635`:

```python
from latex2code import compile_latex

density = compile_latex(
    r"\frac{1}{\sigma \sqrt{2 \pi}}"
    r"\exp(-\frac{(x-\mu)^2}{2 \sigma^2})"
)
value = density(x=0.0, mu=0.0, sigma=2.0)
print(value)  # 0.19947114020071635
```

The expression does not enforce `sigma > 0`; validate model parameters in the calling
application. To evaluate arrays/tensors, select `backend="numpy"`, `"torch"`, or `"jax"` and
choose the input dtype to suit the calculation's precision requirements.

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

Select an output backend with `--backend`:

```powershell
latex2code "\sin(x)^2 + x" --backend torch --output differentiable.py
latex2code "\sin(x)^2 + x" --backend jax --output differentiable.py
```

`--numpy` remains available as a legacy alias for `--backend numpy`. Generated source for
PyTorch/JAX imports the selected framework, so install that framework in the environment where the
generated file will run.
Use `--inspect` to print a JSON report with the parsed expression, function variables, warnings, and source.
Rename parsed symbols with repeatable `--map-variable SYMBOL=NAME` options:

```powershell
latex2code "\theta + x" --map-variable theta=angle
```

Start an interactive session to paste and transpile expressions. Use `--inspect` to show the
structured JSON report for each expression; `quit` or `exit` ends the session. For multiline
expressions, enter `:begin`, paste the expression, then enter `:end`:

```powershell
latex2code --interactive
latex2code --interactive --inspect --backend numpy
```

### Options

| Flag | Long Option | Description |
| --- | --- | --- |
| `-n` | `--name` | Name of the generated Python function (default: `formula`). |
|  | `--no-types` | Disable type hints in the generated signature. |
| `-p` | `--numpy` | Force generation with NumPy functions/arrays. |
|  | `--backend` | Select `python`, `numpy`, `torch`, or `jax` output (default: `python`). |
| `-o` | `--output` | Write generated Python source to a file instead of stdout. |
|  | `--inspect` | Print JSON containing the parsed expression, variables, warnings, and generated source. |
|  | `--map-variable SYMBOL=NAME` | Rename a parsed variable; may be repeated. |
| `-i` | `--interactive` | Read and transpile expressions until `quit`, `exit`, or end-of-file. |

---

## Running Tests

Run the test suite using `pytest`:

```powershell
pytest
```

## Supported Scope and Limitations

- The table describes the supported shapes of input; it does not claim full TeX macro or document parsing.

| Area | Supported examples | Notes |
| --- | --- | --- |
| Arithmetic | `+`, `-`, products, fractions, powers, roots, nth roots | Parsed expressions follow SymPy's supported LaTeX grammar. |
| Functions | Trigonometric, inverse/reciprocal trig, hyperbolic, exponential/logarithmic, `erf`, `gamma`, `beta`, `max`, `min` | Function support can depend on backend printer support; beta is generated using log-gamma arithmetic. |
| ML operators | `sigmoid`, `relu`, `softplus`, `logit`, `softsign`, `swish` | Use `\operatorname{...}(x)`; sigmoid, softplus, and swish use stable scalar/framework-native implementations. |
| Symbolic calculus | Derivatives, limits, integrals, finite sums and reducible finite products | SymPy evaluates operations when possible; unresolved products fail rather than emitting incomplete code. |
| Arrays and cases | Matrix/array layouts, `cases`, selected aligned equation layouts | Aligned equations are represented as rows of values, not solved as systems. |
| Inspection and names | Parsed expression tree and variable-to-identifier mapping | Mappings rename arguments only; they do not substitute values or change expression semantics. |

Backend behavior below describes tested pathways, not a guarantee that every parser-supported
function has equivalent semantics across targets:

| Capability | Python | NumPy | PyTorch | JAX |
| --- | --- | --- | --- | --- |
| Scalar arithmetic and common functions | Yes | Yes | Yes, for generated native operations | Yes, for generated native operations |
| Elementwise arrays/tensors | No array contract | NumPy broadcasting | Pass tensors with desired dtype/device | Pass arrays with desired dtype |
| Matrix/array constructors | NumPy output for compatibility | NumPy arrays | Tensor output; gradients through dynamic entries | JAX arrays |
| Piecewise array evaluation | Python conditional expression | `where`, eager branches | `where`, eager branches | `where`, eager branches |
| Automatic differentiation | No | No | Yes, for differentiable native operations | Yes, for differentiable native operations |
| JIT compilation | No library guarantee | No library guarantee | No library guarantee | `jax.jit` tested for supported primitives |

PyTorch and JAX are optional runtime dependencies. Operations without a native differentiable
implementation are rejected rather than silently replaced with Python math. JAX commonly defaults
to 32-bit floating point; enable and verify 64-bit mode in the application when needed. Python and
NumPy error/gamma functions can use Python `math` implementations wrapped for arrays, which may be
slower than native array kernels.

- Input is parsed strictly by SymPy's LaTeX parser plus a restricted safe fallback, so malformed trailing tokens are rejected rather than silently discarded. Document markup, arbitrary TeX macros, and arbitrary custom functions are outside scope.
- Presentation-only spacing commands such as `\quad`, `\qquad`, `\,`, `\hspace{...}`, and `\vspace{...}` are ignored. `\displaystyle` and related math-style commands are also ignored. Text wrappers and unknown macros are not stripped because they can carry mathematical meaning; remove or rewrite them explicitly if unsupported.
- Function application and implicit multiplication follow the parser's documented grammar; no assumptions are made about custom function names or subscript semantics. Inspect the parsed expression/tree and explicitly rename variables instead of relying on guessed intent.
- Unsupported commands and functions raise `UnsupportedLaTeXFeatureError`; close command/function spellings are suggested when possible. Malformed input raises `InvalidLaTeXSyntaxError`, which includes a source location and excerpt when identifiable, plus delimiter or environment hints when detected.
- For example, a misspelled `\sinn(x)` reports the likely `\sin` command, while a fraction with an unmatched brace points out the unbalanced delimiters.
- Piecewise cases without an explicit default remain undefined outside their listed conditions; generated Python returns `None` there, while NumPy mode represents those values as `numpy.nan`.
- `\pi` and `e` are numeric constants; other Greek symbols may become function parameters. Use `inspect_latex` or CLI `--inspect` to review parsed variables and generated code.
- Derivatives and integrals are evaluated symbolically when SymPy can do so. An indefinite integral returns one antiderivative and does not add the arbitrary constant of integration.
- Finite sums are emitted as executable Python sum expressions. Matrices use the selected array backend (Python's matrix output remains a NumPy array for compatibility).
- Finite products are symbolically evaluated when SymPy can reduce them; unresolved symbolic products raise `CodeGenerationError` rather than returning incomplete Python.
- NumPy, PyTorch, and JAX operate elementwise where the framework supports the operation. Finite sums still use scalar iteration bounds.
- Piecewise outputs use elementwise selection in array backends. Validate branch domains when an unselected branch could itself produce invalid values.
- Generated tensor code does not promise a particular device, dtype, or shape policy. Gamma has poles at non-positive integers; behavior at singularities and outside mathematical domains follows the selected backend.
- Generated Python is not a mathematical proof or a guarantee of numerical stability. Review and test output before relying on it in research or production calculations.

---

## Scientific Usage

```python
import numpy as np
from latex2code import latex_to_code

# Convert LaTeX formula directly to NumPy executable string
latex_eq = r"\frac{\sin(x)}{\sqrt{x^2 + 1}}"
python_code = latex_to_code(latex_eq)

x = np.linspace(0, 10, 100)
result = eval(python_code)
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
