"""A short tour of latex2code.

Run with:  python examples/basic_usage.py
Every formula here also appears in the README or the test suite.
"""

import numpy as np

from latex2code import compile_latex, inspect_latex, transpile_latex


def show(title: str, text: str) -> None:
    print(f"\n=== {title} ===\n{text}")


# 1. Source code you can read, with the LaTeX recorded in the docstring.
show(
    "Generated source",
    transpile_latex(
        r"\frac{\sin(x) + \cos(y)}{\sqrt{z}}",
        function_name="calculate_wave",
        docstring=True,
    ),
)

# 2. A callable you can use right away. Gaussian density at the mean, sigma = 2.
density = compile_latex(
    r"\frac{1}{\sigma \sqrt{2 \pi}}\exp(-\frac{(x-\mu)^2}{2 \sigma^2})"
)
show("Gaussian density", f"{density(x=0.0, mu=0.0, sigma=2.0)!r}  (expected 0.19947114020071635)")

# 3. Vectorized evaluation with the NumPy backend.
sinc_like = compile_latex(r"\frac{\sin(x)}{\sqrt{x^2 + 1}}", backend="numpy")
show("NumPy backend", str(sinc_like(np.linspace(0, 10, 5))))

# 4. Piecewise functions from \begin{cases}.
absolute = compile_latex(
    r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}",
    function_name="abs_value",
    type_hints=False,
)
show("Piecewise", f"abs_value(-3) = {absolute(-3)}")

# 5. Rename arguments without changing the mathematics.
model = compile_latex(
    r"\theta + \lambda",
    variable_map={"theta": "angle", "lambda": "wavelength"},
)
show("variable_map", f"model(angle=0.5, wavelength=2.0) = {model(angle=0.5, wavelength=2.0)}")

# 6. Check how an expression was interpreted before trusting it.
details = inspect_latex(r"\sin(\pi x) + \theta")
show(
    "Inspection",
    f"parsed:    {details.parsed_expression}\n"
    f"variables: {details.variables}\n"
    f"warnings:  {details.warnings}",
)

# 7. Exact integer binomials on the Python backend.
choose = compile_latex(r"\binom{n}{k}", function_name="choose", type_hints=False)
show("Binomial", f"choose(60, 30) = {choose(60, 30)}")
