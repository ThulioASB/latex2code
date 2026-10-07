from io import StringIO
import math
import numpy as np
import pytest
from latex2code import transpile_latex
from latex2code.core import (
    CodeGenerationError,
    InvalidLaTeXSyntaxError,
    InvalidPythonIdentifierError,
    LaTeXTranspilerError,
    UnsupportedLaTeXFeatureError,
)
from latex2code.cli import main as cli_main


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
    scope = {}
    exec(code, scope)
    result = scope["formula"](np.array([0.0, math.pi / 2]), np.array([0.0, 0.0]))
    assert np.allclose(result, np.array([1.0, 2.0]))


def test_finite_product_is_emitted_as_executable_python():
    code = transpile_latex(r"\prod_{i=1}^{n} i", type_hints=False)
    scope = {}
    exec(code, scope)
    assert scope["formula"](5) == 120


def test_unresolved_product_raises_a_clear_codegen_error():
    with pytest.raises(CodeGenerationError, match="Unable to generate Python code"):
        transpile_latex(r"\prod_{i=1}^{n} \sin(i)", type_hints=False)


@pytest.mark.parametrize(
    ("latex", "values", "expected"),
    [
        (r"\left|x\right|", (np.array([-2.0, 3.0]),), np.array([2.0, 3.0])),
        (r"\max(x, y)", (np.array([1.0, 4.0]), np.array([2.0, 3.0])), np.array([2.0, 4.0])),
        (
            r"\max(x, y, z)",
            (np.array([1.0, 4.0]), np.array([2.0, 3.0]), np.array([0.0, 5.0])),
            np.array([2.0, 5.0]),
        ),
        (r"\operatorname{erf}(x)", (np.array([0.0, 1.0]),), np.array([0.0, math.erf(1.0)])),
    ],
)
def test_numpy_mode_vectorizes_math_functions(latex, values, expected):
    code = transpile_latex(latex, use_numpy=True, type_hints=False)
    scope = {}
    exec(code, scope)
    assert np.allclose(scope["formula"](*values), expected)


def test_numpy_piecewise_works_with_array_inputs():
    code = transpile_latex(
        r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}",
        use_numpy=True,
        type_hints=False,
    )
    scope = {}
    exec(code, scope)
    result = scope["formula"](np.array([-2.0, 0.0, 3.0]))
    assert np.allclose(result, np.array([2.0, 0.0, 3.0]))


def test_matrix_transpilation():
    latex_matrix = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
    code = transpile_latex(latex_matrix, use_numpy=True)
    scope = {"np": np}
    exec(code, scope)
    result = scope["formula"](1, 2, 3, 4)
    expected = np.array([[1, 2], [3, 4]])
    assert np.array_equal(result, expected)


def test_type_hints_match_matrix_inputs_and_return_type():
    code = transpile_latex(r"\begin{pmatrix} x & 1 \\ 0 & y \end{pmatrix}")
    assert "def formula(x: float, y: float) -> np.ndarray:" in code


def test_numpy_matrix_hints_preserve_array_return_type():
    code = transpile_latex(r"\begin{pmatrix} x & 1 \\ 0 & y \end{pmatrix}", use_numpy=True)
    assert "def formula(x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:" in code


def test_numpy_type_hints_allow_scalar_or_array_inputs_and_results():
    code = transpile_latex(r"\sin(x)", use_numpy=True)
    assert "def formula(x: float | np.ndarray) -> float | np.ndarray:" in code


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


def test_piecewise_expression_matches_expected_value():
    code = transpile_latex(r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}", type_hints=False)
    scope = {}
    exec(code, scope)
    assert scope["formula"](3) == 3
    assert scope["formula"](-2) == 2


def test_piecewise_expression_preserves_undefined_regions():
    code = transpile_latex(r"\begin{cases} x & x > 0 \\ -x & x < 0 \end{cases}", type_hints=False)
    scope = {}
    exec(code, scope)
    assert scope["formula"](2) == 2
    assert scope["formula"](-2) == 2
    assert scope["formula"](0) is None


def test_special_functions_and_operatorname_are_supported():
    code = transpile_latex(r"\operatorname{erf}(x) + \sinh(y)", type_hints=False)
    scope = {}
    exec(code, scope)
    assert math.isclose(scope["formula"](0.5, 0.25), math.erf(0.5) + math.sinh(0.25))


def test_operatorname_does_not_collide_with_existing_symbols():
    code = transpile_latex(r"\operatorname{erf}(x) + A", type_hints=False)
    scope = {}
    exec(code, scope)
    assert math.isclose(scope["formula"](2, 0.5), math.erf(0.5) + 2)


def test_limits_binomials_and_abs_are_supported():
    limit_code = transpile_latex(r"\lim_{x \to 0} \frac{\sin(x)}{x}", type_hints=False)
    limit_scope = {}
    exec(limit_code, limit_scope)
    assert math.isclose(limit_scope["formula"](), 1.0)

    binomial_code = transpile_latex(r"\binom{n}{k}", type_hints=False)
    binomial_scope = {}
    exec(binomial_code, binomial_scope)
    assert binomial_scope["formula"](5, 2) == 10

    abs_code = transpile_latex(r"\left|x\right|", type_hints=False)
    abs_scope = {}
    exec(abs_code, abs_scope)
    assert abs_scope["formula"](-3) == 3


def test_unsupported_commands_and_functions_have_specific_errors():
    with pytest.raises(UnsupportedLaTeXFeatureError, match=r"Unsupported LaTeX command: \\unknown"):
        transpile_latex(r"\unknown(x)")
    with pytest.raises(UnsupportedLaTeXFeatureError, match="Unsupported function: custom"):
        transpile_latex(r"\operatorname{custom}(x)")


def test_safe_fallback_rejects_non_math_python_syntax():
    with pytest.raises(InvalidLaTeXSyntaxError):
        transpile_latex(r"\mathrm{erf}(__import__(x))")


def test_matrix_rows_must_have_matching_dimensions():
    with pytest.raises(InvalidLaTeXSyntaxError):
        transpile_latex(r"\begin{pmatrix} x & y \\ z \end{pmatrix}")


@pytest.mark.parametrize(
    "latex",
    [
        r"\begin{pmatrix} x & y \\ z & w \end{bmatrix}",
        r"\begin{pmatrix} x & y \\ z & w \end{pmatrix} + 1",
    ],
)
def test_matrix_parser_rejects_mismatched_or_trailing_input(latex):
    with pytest.raises(InvalidLaTeXSyntaxError):
        transpile_latex(latex)


def test_cases_parser_rejects_trailing_input():
    with pytest.raises(InvalidLaTeXSyntaxError):
        transpile_latex(r"\begin{cases} x & x > 0 \end{cases} + 1")


def test_cli_reads_latex_from_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["latex2code", "-", "--name", "stdin_formula"])
    monkeypatch.setattr("sys.stdin", StringIO(r"\frac{x}{2}"))

    cli_main()

    output = capsys.readouterr().out
    assert "def stdin_formula(x: float) -> float:" in output
    assert "return x/2" in output


def test_cli_writes_generated_source_to_file(monkeypatch, capsys, tmp_path):
    output_file = tmp_path / "generated.py"
    monkeypatch.setattr(
        "sys.argv",
        ["latex2code", r"\frac{x}{2}", "--name", "half", "--output", str(output_file)],
    )

    cli_main()

    assert output_file.read_text(encoding="utf-8") == (
        "def half(x: float) -> float:\n    return x/2\n"
    )
    assert capsys.readouterr().out == ""


def test_cli_reports_output_write_errors(monkeypatch, capsys, tmp_path):
    output_file = tmp_path / "missing" / "generated.py"
    monkeypatch.setattr(
        "sys.argv",
        ["latex2code", "x", "--output", str(output_file)],
    )

    with pytest.raises(SystemExit) as error:
        cli_main()

    assert error.value.code == 1
    assert "Could not write output" in capsys.readouterr().err


def test_cli_reports_invalid_input_with_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["latex2code", r"\unknown(x)"])

    with pytest.raises(SystemExit) as error:
        cli_main()

    assert error.value.code == 1
    assert "Unsupported LaTeX command: \\unknown" in capsys.readouterr().err


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