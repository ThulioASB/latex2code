# Usage guide

This guide holds the detailed examples that used to live in the README. For a quick start, see
the [README](../README.md).

## Choosing between `transpile_latex`, `compile_latex`, and `inspect_latex`

| Function | Returns | Use it when |
| --- | --- | --- |
| `transpile_latex` | Python source (`str`) | You want to write the code to a file or review it. |
| `compile_latex` | `CompiledFormula` (callable, with `.source` and `.info`) | You want to evaluate the formula right away, for example in a notebook. |
| `inspect_latex` | `TranspilationInfo` | You want to check how the expression was interpreted. |

All three accept the same keyword-only options: `function_name`, `type_hints`, `docstring`,
`backend`, `variable_map`, `args`, and `constants`.

`compile_latex` needs the optional framework installed when you select `torch` or `jax`. The
callable uses only the arguments you pass and does not move or cast array/tensor inputs.

## Constants, Greek variables, and argument order

```python
from latex2code import transpile_latex

print(transpile_latex(r"\sin(\pi x) + e + \theta", function_name="signal_response"))
```

`\pi` and `e` are emitted as numeric constants by default. Other Greek letters such as
`\theta` remain variables and appear in the function signature.

- Pass `constants` to customize substitutions. For example `constants={"pi": math.pi}` leaves
  `e` as a function argument.
- Pass `args=["x", "w", "b"]` to set argument order explicitly. The default follows each
  variable's first appearance in the input.
- Text wrappers keep their contents: `\mathrm{e}^x` retains the `e`, and `\text{...}` is not
  removed as if it were whitespace.

## Renaming arguments

`variable_map` gives parsed symbols clearer Python names without changing the mathematics.
Keys are parsed symbol names and values must be unique Python identifiers:

```python
from latex2code import compile_latex

model = compile_latex(
    r"\theta + \lambda",
    variable_map={"theta": "angle", "lambda": "wavelength"},
)
print(model(angle=0.5, wavelength=2.0))
```

From the CLI: `latex2code '\theta + x' --map-variable theta=angle`.

## Recording the source LaTeX

Generated functions can carry the LaTeX they came from, so output stays traceable to the paper
it was transcribed from:

```python
from latex2code import transpile_latex

print(transpile_latex(r"\frac{x}{y}", docstring=True))
```

```python
def formula(x: float, y: float) -> float:
    r"""LaTeX source:
        \frac{x}{y}
    """
    return x/y
```

The docstring is off by default. The LaTeX is escaped so it cannot terminate the string early;
unusual input (embedded triple quotes, control characters) falls back to a single-line
`repr`-style literal.

## Inspecting a transpilation

```python
from latex2code import inspect_latex

details = inspect_latex(r"\sin(\pi x) + \theta")
print(details.input_expression)
print(details.backend)
print(details.parsed_expression)   # readable SymPy form
print(details.expression_tree)     # JSON-compatible tree of operations and arguments
print(details.variables)
print(details.generated_code)
print(details.warnings)
```

`latex2code '\sin(\pi x) + \theta' --inspect` prints the same information as JSON. Parse and
unsupported-feature errors include a line and column with a short source excerpt when the
location can be identified.

## Matrices

```python
from latex2code import transpile_latex

latex_matrix = r"\begin{pmatrix} x^2 & 1 \\ 0 & y \end{pmatrix}"
print(transpile_latex(latex_matrix, function_name="create_matrix", type_hints=False))
```

```python
import numpy as np

def create_matrix(x, y):
    return np.array([[x**2, 1], [0, y]])
```

Matrices use the selected array backend. The `python` backend still emits a NumPy array for
compatibility.

## Piecewise expressions

```python
from latex2code import transpile_latex

print(
    transpile_latex(
        r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}",
        function_name="abs_value",
        type_hints=False,
    )
)
```

```python
def abs_value(x):
    return (x) if (x > 0) else (-x)
```

Cases without an explicit default are undefined outside their listed conditions: generated
Python returns `None` there, and NumPy mode produces `numpy.nan`.

On array backends, piecewise functions use eager elementwise selection (`where`): every branch
may be evaluated before selection. Avoid invalid operations in inactive branches (for example
`log(x)` in a branch where `x <= 0`), or write a framework-specific implementation if you need
true lazy control flow. Transpiling piecewise expressions for NumPy, PyTorch, or JAX emits a
`PiecewiseEvaluationWarning`. Automatic domain clipping is intentionally not performed because
it changes the formula and its gradients.

## Machine-learning operators

Activation and link functions are written with `\operatorname{...}`:

```python
from latex2code import transpile_latex

print(
    transpile_latex(
        r"\operatorname{sigmoid}(x)",
        function_name="logistic_probability",
        backend="numpy",
    )
)
```

Supported: `sigmoid`, `relu`, `softplus`, `softmax`, `gelu`, `silu`, `logit`, `softsign`,
`swish`.

Named `sigmoid`, `softplus`, and `swish` operators use numerically stable scalar or
framework-native implementations. Writing an equivalent expanded formula by hand (such as
`log(1 + exp(x))`) does not get that stabilization. `softmax`, `gelu`, and `silu` are
generated from their standard definitions; see [backends.md](backends.md) for the details per
backend. Check input domains for `logit`, logarithms, and roots: the transpiler does not
silently clip inputs.

## Using generated code in training or analysis

Choose tensor dtype, device, and shape in your application. Generated functions preserve
framework operations and do not move or silently cast tensor inputs:

```python
import torch
from latex2code import transpile_latex

source = transpile_latex(r"\operatorname{softplus}(x) + x^2", backend="torch")
namespace = {}
exec(source, namespace)  # only do this with LaTeX you trust
formula = namespace["formula"]

x = torch.tensor([0.25, 1.0], dtype=torch.float64, requires_grad=True)
loss = formula(x).mean()
loss.backward()
```

For JAX, generated functions can be passed to `jax.jit`, `jax.vmap`, and `jax.grad` when the
expression uses supported JAX primitives. Test the actual shapes and dtypes your workload uses.

## Worked example: Gaussian density

The normal probability density uses `x`, `mu`, and `sigma` in the same units; the output has
inverse units. Require `sigma > 0`. At the mean with `sigma = 2` the density is
`1 / (2 sqrt(2 pi))`, approximately `0.19947114020071635`:

```python
from latex2code import compile_latex

density = compile_latex(
    r"\frac{1}{\sigma \sqrt{2 \pi}}"
    r"\exp(-\frac{(x-\mu)^2}{2 \sigma^2})"
)
print(density(x=0.0, mu=0.0, sigma=2.0))  # 0.19947114020071635
```

The expression does not enforce `sigma > 0`; validate model parameters in the calling
application. For arrays or tensors, select `backend="numpy"`, `"torch"`, or `"jax"` and choose
the input dtype to suit your precision requirements.

## Limits, binomials, and products

```python
from latex2code import transpile_latex

print(transpile_latex(r"\lim_{x \to 0} \frac{\sin(x)}{x}", function_name="sinc_limit", type_hints=False))
print(transpile_latex(r"\binom{n}{k}", function_name="choose", type_hints=False))
print(transpile_latex(r"\prod_{i=1}^{n} i", function_name="factorial_product", type_hints=False))
```

```python
def sinc_limit():
    return 1

import math

def choose(n, k):
    return (math.comb(n, k) if isinstance(n, int) and isinstance(k, int) and n >= 0 and k >= 0 else math.gamma(n + 1)/(math.gamma(k + 1)*math.gamma(-k + n + 1)))

import math

def factorial_product(n):
    return math.factorial(n)
```

On the `python` backend, `\binom{n}{k}` uses `math.comb` when both arguments are non-negative
Python integers, so large coefficients are exact, and falls back to a gamma-function quotient
for real-valued arguments.
