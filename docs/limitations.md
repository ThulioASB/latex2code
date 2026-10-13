# Supported scope and limitations

latex2code transpiles a documented subset of LaTeX math. It does not parse TeX documents or
arbitrary macros.

## Supported input

| Area | Supported examples | Notes |
| --- | --- | --- |
| Arithmetic | `+`, `-`, products, fractions, powers, roots, nth roots | Parsed expressions follow SymPy's supported LaTeX grammar. |
| Functions | Trigonometric, inverse/reciprocal trig, hyperbolic, exponential/logarithmic, `erf`, `gamma`, `beta`, `max`, `min` | Support can depend on backend printer support; `beta` is generated using log-gamma arithmetic. |
| ML operators | `sigmoid`, `relu`, `softplus`, `softmax`, `gelu`, `silu`, `logit`, `softsign`, `swish` | Use `\operatorname{...}(x)`. See [backends.md](backends.md) for per-backend behavior and stability. |
| Symbolic calculus | Derivatives, limits, integrals, finite sums, reducible finite products | SymPy evaluates operations when possible; unresolved products fail rather than emitting incomplete code. |
| Arrays and cases | Matrix/array layouts, `cases`, selected aligned equation layouts | Aligned equations are represented as rows of values, not solved as systems. |
| Inspection and names | Parsed expression tree and variable-to-identifier mapping | Mappings rename arguments only; they do not substitute values or change semantics. |

## Parsing

- Input is parsed strictly by SymPy's ANTLR-based LaTeX parser plus a restricted safe fallback,
  so malformed trailing tokens are rejected rather than silently discarded. Document markup,
  arbitrary TeX macros, and arbitrary custom functions are out of scope.
- SymPy's LaTeX grammar requires the `antlr4-python3-runtime` package, pinned to the 4.11.x
  series that SymPy's generated parser expects. SymPy's Lark backend was evaluated but does not
  cover the expression grammar and constructs supported here.
- Presentation-only spacing commands such as `\quad`, `\qquad`, `\,`, `\hspace{...}`, and
  `\vspace{...}` are ignored, as are `\displaystyle` and related math-style commands. Text
  wrappers and unknown macros are not stripped because they can carry mathematical meaning;
  remove or rewrite them explicitly if unsupported.
- Function application and implicit multiplication follow the parser's documented grammar. No
  assumptions are made about custom function names or subscript semantics. Inspect the parsed
  expression or tree and rename variables explicitly rather than relying on guessed intent.

## Errors

- Unsupported commands and functions raise `UnsupportedLaTeXFeatureError`; close spellings are
  suggested when possible. For example, a misspelled `\sinn(x)` reports the likely `\sin`.
- Malformed input raises `InvalidLaTeXSyntaxError`, which includes a source location and
  excerpt when identifiable, plus delimiter or environment hints when detected. For example, a
  fraction with an unmatched brace points out the unbalanced delimiters.

## Semantics worth knowing

- `\pi` and `e` are numeric constants; other Greek symbols may become function parameters. Use
  `inspect_latex` or `--inspect` to review parsed variables and generated code.
- Derivatives and integrals are evaluated symbolically when SymPy can do so. An indefinite
  integral returns one antiderivative and does not add the arbitrary constant of integration.
- Finite sums are emitted as executable Python sum expressions, and still use scalar iteration
  bounds on array backends. Finite products are evaluated symbolically when SymPy can reduce
  them; unresolved products raise `CodeGenerationError` rather than returning incomplete code.
- Piecewise cases without an explicit default are undefined outside their listed conditions:
  generated Python returns `None` there and NumPy mode produces `numpy.nan`. On array backends,
  every branch may be evaluated before selection, so keep all branches valid over the full input
  domain.
- NumPy, PyTorch, and JAX operate elementwise where the framework supports the operation.
- Generated tensor code does not promise a particular device, dtype, or shape policy. Gamma has
  poles at non-positive integers; behavior at singularities and outside mathematical domains
  follows the selected backend.
- Generated Python is not a proof of correctness or numerical stability. Review and test output
  before relying on it in research or production.
