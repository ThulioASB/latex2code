import math
import numpy as np
import pytest
from latex2code import transpile_latex
from latex2code.core import (
    InvalidLaTeXSyntaxError,
    InvalidPythonIdentifierError,
    LaTeXTranspilerError,
)


def test_simple_fraction():
    code = transpile_latex(r"\frac{x}{y}")
    scope = {}
    exec(code, scope)
    assert scope["formula"](10, 2) == 5.0


def test_greek_symbols_with_explicit_numeric_values():
    code = transpile_latex(r"\pi \cdot \theta", type_hints=False)
    scope = {}
    exec(code, scope)
    assert math.isclose(scope["formula"](math.pi, 2.0), 2.0 * math.pi)


def test_trigonometric_expression_matches_expected_value():
    code = transpile_latex(r"\sin(\pi x) + \theta", type_hints=False)
    scope = {}
    exec(code, scope)
    result = scope["formula"](math.pi, 2.0, 0.5)
    assert math.isclose(result, math.sin(math.pi * 0.5) + 2.0)


def test_nested_functions():
    code = transpile_latex(r"\sin(\cos(x))", type_hints=False)
    scope = {"math": math}
    exec(code, scope)
    assert math.isclose(scope["formula"](0), math.sin(1.0))


def test_numpy_flag_consistency():
    code = transpile_latex(r"\sin(x) + \cos(y)", use_numpy=True)
    assert "import numpy as np" in code
    assert "math.sin" not in code


def test_matrix_transpilation():
    latex_matrix = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
    code = transpile_latex(latex_matrix, use_numpy=True)
    scope = {"np": np}
    exec(code, scope)
    result = scope["formula"](1, 2, 3, 4)
    expected = np.array([[1, 2], [3, 4]])
    assert np.array_equal(result, expected)


def test_derivative_and_integral():
    latex_diff = r"\frac{d}{dx}(x^3)"
    code = transpile_latex(latex_diff, type_hints=False)
    scope = {}
    exec(code, scope)
    assert scope["formula"](2) == 12


def test_symbolic_indefinite_integral_matches_expected_value():
    code = transpile_latex(r"\int x^2 dx", type_hints=False)
    scope = {}
    exec(code, scope)
    assert math.isclose(scope["formula"](3), 9.0)


def test_summation_matches_expected_value():
    code = transpile_latex(r"\sum_{i=1}^{n} i^2", type_hints=False)
    scope = {}
    exec(code, scope)
    assert scope["formula"](3) == 14


@pytest.mark.parametrize("function_name", ["not-valid", "class", "formula\npass"])
def test_invalid_function_name_is_rejected(function_name):
    with pytest.raises(InvalidPythonIdentifierError, match="Function name"):
        transpile_latex("x", function_name=function_name)


def test_invalid_generated_argument_name_is_rejected():
    with pytest.raises(InvalidPythonIdentifierError, match="Generated argument name"):
        transpile_latex(r"\lambda + x", type_hints=False)


def test_empty_string_error():
    with pytest.raises(LaTeXTranspilerError):
        transpile_latex("")


def test_invalid_syntax():
    with pytest.raises(InvalidLaTeXSyntaxError):
        transpile_latex(r"\frac{x}{2")