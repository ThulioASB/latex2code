import math
import numpy as np
import pytest
import subprocess
import sys
from latex2code import transpile_latex
from latex2code.core import InvalidLaTeXSyntaxError, LaTeXTranspilerError


# ---------------------------------------------------------------------------
# 1. Basic Arithmetic & Functions
# ---------------------------------------------------------------------------

def test_simple_fraction():
    code = transpile_latex(r"\frac{x}{y}")
    scope = {}
    exec(code, scope)
    assert scope["formula"](10, 2) == 5.0


def test_powers_and_polynomials():
    # Tests x^3 + 2x^2 - 5x + 7
    code = transpile_latex(r"x^3 + 2 x^2 - 5 x + 7", type_hints=False)
    scope = {}
    exec(code, scope)
    assert scope["formula"](2) == 13  # 8 + 8 - 10 + 7


def test_nested_fractions():
    # Tests \frac{\frac{a}{b}}{c}
    code = transpile_latex(r"\frac{\frac{a}{b}}{c}")
    scope = {}
    exec(code, scope)
    assert scope["formula"](12, 2, 3) == 2.0


# ---------------------------------------------------------------------------
# 2. Trigonometry & Advanced Math
# ---------------------------------------------------------------------------

def test_trigonometric_identities():
    # sin^2(x) + cos^2(x) should always equal 1
    code = transpile_latex(r"\sin(x)^2 + \cos(x)^2", type_hints=False)
    scope = {"math": math}
    exec(code, scope)
    assert math.isclose(scope["formula"](1.57), 1.0)


def test_square_root_and_exponent():
    # \sqrt{x} + \exp(y)
    code = transpile_latex(r"\sqrt{x} + \exp(y)", type_hints=False)
    scope = {"math": math}
    exec(code, scope)
    # sqrt(9) + exp(0) = 3 + 1 = 4.0
    assert scope["formula"](9, 0) == 4.0


# ---------------------------------------------------------------------------
# 3. Summations & Iterative Logic
# ---------------------------------------------------------------------------

def test_summation_expression():
    # \sum_{i=1}^{n} i^2
    code = transpile_latex(r"\sum_{i=1}^{n} i^2", type_hints=False)
    import builtins
    scope = {"builtins": builtins}
    exec(code, scope)
    assert scope["formula"](3) == 14


# ---------------------------------------------------------------------------
# 4. Matrices & NumPy Integration
# ---------------------------------------------------------------------------

def test_matrix_transpilation():
    latex_matrix = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
    code = transpile_latex(latex_matrix, use_numpy=True)
    scope = {"np": np}
    exec(code, scope)
    result = scope["formula"](1, 2, 3, 4)
    expected = np.array([[1, 2], [3, 4]])
    assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# 5. Type Hints & Custom Function Names
# ---------------------------------------------------------------------------

def test_custom_function_name_and_type_hints():
    code = transpile_latex(r"\frac{a}{b}", function_name="custom_divider", type_hints=True)
    assert "def custom_divider(a: float, b: float) -> float:" in code


def test_disable_type_hints():
    code = transpile_latex(r"a + b", type_hints=False)
    assert "def formula(a, b):" in code


# ---------------------------------------------------------------------------
# 6. Error Handling & Edge Cases
# ---------------------------------------------------------------------------

def test_empty_string_error():
    with pytest.raises(LaTeXTranspilerError):
        transpile_latex("")


def test_invalid_syntax_unclosed_bracket():
    with pytest.raises(InvalidLaTeXSyntaxError):
        transpile_latex(r"\frac{x}{2")


def test_invalid_command_error():
    with pytest.raises(InvalidLaTeXSyntaxError):
        transpile_latex(r"\someUnknownCommand{x}")

def test_derivative_transpilation():
    # Derivative of x^3 with respect to x is 3*x^2
    latex_diff = r"\frac{d}{dx}(x^3)"
    code = transpile_latex(latex_diff, type_hints=False)
    
    scope = {}
    exec(code, scope)
    formula = scope["formula"]
    
    # 3 * (2^2) = 12
    assert formula(2) == 12


def test_integral_transpilation():
    # Indefinite integral of x^2 is (x^3)/3
    latex_int = r"\int x^2 dx"
    code = transpile_latex(latex_int, type_hints=False)
    
    scope = {}
    exec(code, scope)
    formula = scope["formula"]
    
    # (3^3)/3 = 9.0
    assert formula(3) == 9.0

def test_cli_simple_expression():
    result = subprocess.run(
        [sys.executable, "-m", "latex2code.cli", r"\frac{x}{2}", "--name", "half"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "def half(x: float) -> float:" in result.stdout


def test_cli_invalid_expression():
    result = subprocess.run(
        [sys.executable, "-m", "latex2code.cli", r"\frac{x}{"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Error" in result.stderr