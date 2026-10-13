# Backends

Select a backend with `backend="python" | "numpy" | "torch" | "jax"` (library) or
`--backend` (CLI). The default is `python`.

The table describes tested pathways. It is not a guarantee that every parser-supported
function has equivalent semantics on every target.

| Capability | Python | NumPy | PyTorch | JAX |
| --- | --- | --- | --- | --- |
| Scalar arithmetic and common functions | Yes | Yes | Yes, for generated native operations | Yes, for generated native operations |
| Elementwise arrays/tensors | No array contract | NumPy broadcasting | Pass tensors with the desired dtype/device | Pass arrays with the desired dtype |
| Matrix/array constructors | NumPy output, for compatibility | NumPy arrays | Tensor output; gradients flow through dynamic entries | JAX arrays |
| Piecewise evaluation | Python conditional expression | `where`, eager branches | `where`, eager branches | `where`, eager branches |
| Automatic differentiation | No | No | Yes, for differentiable native operations | Yes, for differentiable native operations |
| JIT compilation | No library guarantee | No library guarantee | No library guarantee | `jax.jit` tested for supported primitives |
| Generated-code imports | `math` (plus `numpy` for matrices) | `numpy` (plus `math` when a formula uses it) | `torch`, `torch.nn.functional` | `jax`, `jax.numpy`, `jax.scipy.special`, `jax.nn` |

PyTorch and JAX are optional runtime dependencies. Operations without a native differentiable
implementation are rejected with a `CodeGenerationError` rather than silently replaced with
Python `math`. JAX commonly defaults to 32-bit floats; enable and verify 64-bit mode in your
application when you need it. On NumPy, `erf`, `gamma`, and `lgamma` use Python `math`
implementations wrapped with `np.vectorize`, which can be slower than native array kernels.
Generated NumPy code needs only NumPy; SciPy is not required.

## Machine-learning operators

| Operator | Python | NumPy | PyTorch | JAX |
| --- | --- | --- | --- | --- |
| `sigmoid` | `0.5 * (1 + tanh(x / 2))` (stable) | `exp(-logaddexp(0, -x))` (stable) | `torch.sigmoid` | `jax.nn.sigmoid` |
| `softplus` | `max(x, 0) + log1p(exp(-abs(x)))` (stable) | `logaddexp(0, x)` (stable) | `F.softplus` | `jax.nn.softplus` |
| `swish` | `x * sigmoid(x)` | `x * sigmoid(x)` | `x * sigmoid(x)` | `x * sigmoid(x)` |
| `silu` | `x / (1 + exp(-x))` | `x / (1 + exp(-x))` | `F.silu` | `jax.nn.silu` |
| `gelu` | exact (`erf`) form | exact (`erf`) form | `F.gelu` (exact) | `jax.nn.gelu(approximate=False)` (exact) |
| `softmax` | list comprehension over the input; no max-subtraction | pure-NumPy, max-subtracted (stable); normalizes over all elements, like `scipy.special.softmax` with its default | `F.softmax(dim=-1)` | `jax.nn.softmax` (last axis) |
| `relu`, `logit`, `softsign` | built from elementary operations (`max`, `log`, `abs`) | same, with `np.maximum` | same, with `torch.maximum` | same, with `jnp.maximum` |

Notes:

- Only `sigmoid`, `softplus`, and `swish` are documented as numerically stable. The Python
  backend's `softmax` is the textbook formula and can overflow for large inputs; use the
  NumPy, PyTorch, or JAX backend for real workloads.
- `softmax` normalization axis differs by backend (see the table). If your formula depends on
  a specific axis, check it with `inspect_latex`.
- Writing an equivalent formula by hand (such as `log(1 + exp(x))`) does not receive these
  implementations.

## Binomial coefficients

| Backend | `\binom{n}{k}` |
| --- | --- |
| Python | `math.comb` for non-negative integer arguments (exact); gamma-function quotient otherwise |
| NumPy | Gamma-function quotient (`np.vectorize(math.gamma)`), works elementwise on arrays |
| PyTorch | Gamma-function quotient built from `torch.lgamma` with a reflection sign term, differentiable |
| JAX | Gamma-function quotient using `jax.scipy.special.gamma` |

Gamma-based forms return floats and can lose precision for large integer arguments.
