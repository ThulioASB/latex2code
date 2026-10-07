import json
from io import StringIO
import math
import numpy as np
import pytest
from latex2code import __version__, compile_latex, inspect_latex, transpile_latex
from latex2code.core import (
    CodeGenerationError,
    InvalidLaTeXSyntaxError,
    InvalidPythonIdentifierError,
    LaTeXTranspilerError,
    PiecewiseEvaluationWarning,
    UnsupportedLaTeXFeatureError,
)
from latex2code.cli import main as cli_main


def test_simple_fraction():
    code = transpile_latex(r"\frac{x}{y}")
    scope = {}
    exec(code, scope)
    assert scope["formula"](10, 2) == 5.0


def test_pi_is_a_constant_and_greek_symbols_are_variables():
    code = transpile_latex(r"\pi \cdot \theta", type_hints=False)
    scope = {}
    exec(code, scope)
    assert "def formula(theta):" in code
    assert math.isclose(scope["formula"](2.0), 2.0 * math.pi)


@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_standard_constants_are_numeric_and_greek_variables_remain_arguments(backend):
    code = transpile_latex(r"\pi + e + \theta", backend=backend, type_hints=False)
    scope = {}
    exec(code, scope)

    assert "def formula(theta):" in code
    assert math.isclose(scope["formula"](2.5), math.pi + math.e + 2.5)


def test_inspection_api_reports_expression_variables_warnings_and_source():
    with pytest.warns(PiecewiseEvaluationWarning):
        details = inspect_latex(
            r"\begin{cases} \pi x & x > 0 \\ e & x \le 0 \end{cases}",
            backend="numpy",
            type_hints=False,
        )

    assert details.parsed_expression.startswith("Piecewise(")
    assert details.input_expression.startswith(r"\begin{cases}")
    assert details.backend == "numpy"
    assert details.variables == ("x",)
    assert len(details.warnings) == 1
    assert "eager elementwise selection" in details.warnings[0]
    assert "def formula(x):" in details.generated_code
    assert "np.pi" in details.generated_code
    assert details.expression_tree["type"] == "Piecewise"
    assert len(details.expression_tree["args"]) == 2


def test_variable_mapping_renames_arguments_and_preserves_values():
    compiled = compile_latex(
        r"\sin(\theta) + \lambda",
        variable_map={"theta": "angle", "lambda": "wavelength"},
    )

    assert compiled.info.variables == ("wavelength", "angle")
    assert "def formula(wavelength: float, angle: float)" in compiled.source
    assert math.isclose(
        compiled(angle=0.5, wavelength=2.0),
        math.sin(0.5) + 2.0,
    )


def test_variable_mapping_works_inside_matrix_outputs():
    compiled = compile_latex(
        r"\begin{pmatrix} \theta & 1 \\ 0 & \lambda \end{pmatrix}",
        backend="numpy",
        type_hints=False,
        variable_map={"theta": "angle", "lambda": "wavelength"},
    )

    assert compiled.info.variables == ("angle", "wavelength")
    assert np.array_equal(
        compiled(angle=2.0, wavelength=3.0),
        np.array([[2.0, 1.0], [0.0, 3.0]]),
    )


@pytest.mark.parametrize(
    ("variable_map", "message"),
    [
        ({"missing": "name"}, "not present in the expression"),
        ({"x": "invalid-name"}, "valid non-keyword Python identifier"),
        ({"x": "result", "y": "result"}, "same Python identifier"),
    ],
)
def test_variable_mapping_rejects_unknown_invalid_or_colliding_names(variable_map, message):
    with pytest.raises(LaTeXTranspilerError, match=message):
        transpile_latex("x + y", variable_map=variable_map)


@pytest.mark.parametrize(
    "spacing",
    [r"\quad", r"\qquad", r"\,", r"\;", r"\!", r"\hspace{1em}", r"\vspace{2pt}"],
)
def test_presentation_spacing_commands_do_not_change_expression(spacing):
    plain = transpile_latex("x + y", type_hints=False)
    spaced = transpile_latex(f"x {spacing} + {spacing} y", type_hints=False)

    assert spaced == plain


@pytest.mark.parametrize("style_command", [r"\displaystyle", r"\textstyle", r"\scriptstyle"])
def test_math_style_commands_do_not_change_expression(style_command):
    assert transpile_latex(f"{style_command} x + y") == transpile_latex("x + y")


def test_variable_mapping_requires_string_names():
    with pytest.raises(LaTeXTranspilerError, match="keys and values must be strings"):
        transpile_latex("x", variable_map={1: "x"})


def test_gaussian_density_example_has_reproducible_reference_value():
    density = compile_latex(
        r"\frac{1}{\sigma \sqrt{2 \pi}}"
        r"\exp(-\frac{(x-\mu)^2}{2 \sigma^2})",
        type_hints=False,
    )

    assert density.info.variables == ("mu", "sigma", "x")
    assert math.isclose(
        density(x=0.0, mu=0.0, sigma=2.0),
        0.19947114020071635,
        rel_tol=1e-14,
    )
    assert math.isclose(
        density(x=2.0, mu=0.0, sigma=2.0),
        math.exp(-0.5) / (2 * math.sqrt(2 * math.pi)),
        rel_tol=1e-14,
    )


def test_compile_api_returns_callable_with_reusable_source_and_metadata():
    compiled = compile_latex(r"\pi x + \theta", function_name="model", type_hints=False)

    assert compiled(theta=0.5, x=2.0) == 2 * math.pi + 0.5
    assert compiled.function(theta=0.5, x=2.0) == compiled(theta=0.5, x=2.0)
    assert compiled.info.variables == ("theta", "x")
    assert compiled.source == compiled.info.generated_code
    assert "def model(theta, x):" in compiled.source


def test_compile_api_supports_numpy_arrays():
    compiled = compile_latex(r"\sin(x) + x^2", backend="numpy", type_hints=False)
    values = np.asarray([0.0, 0.5, 1.0])

    assert np.allclose(compiled(values), np.sin(values) + values**2)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_compile_api_preserves_framework_autodiff(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        compiled = compile_latex(r"\sin(x) + x^2", backend=backend, type_hints=False)
        value = torch.tensor(0.5, requires_grad=True)
        result = compiled(value)
        result.backward()
        assert math.isclose(value.grad.item(), math.cos(0.5) + 1, rel_tol=1e-6)
    else:
        jax = pytest.importorskip("jax")
        jax_numpy = pytest.importorskip("jax.numpy")
        compiled = compile_latex(r"\sin(x) + x^2", backend=backend, type_hints=False)
        value = jax_numpy.asarray(0.5)
        gradient = jax.grad(compiled)(value)
        assert math.isclose(float(gradient), math.cos(0.5) + 1, rel_tol=1e-6)


def test_trigonometric_expression_matches_expected_value():
    code = transpile_latex(r"\sin(\pi x) + \theta", type_hints=False)
    scope = {}
    exec(code, scope)
    result = scope["formula"](2.0, 0.5)
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


@pytest.mark.parametrize("operator", ["erf", "erfc", "gamma"])
@pytest.mark.parametrize("backend", ["python", "numpy"])
def test_python_and_numpy_special_functions_match_reference_values(operator, backend):
    values = [-1.5, -0.5, 0.5, 1.5, 5.0]
    reference = {
        "erf": math.erf,
        "erfc": math.erfc,
        "gamma": math.gamma,
    }[operator]
    expected = [reference(value) for value in values]
    code = transpile_latex(
        rf"\operatorname{{{operator}}}(x)",
        backend=backend,
        type_hints=False,
    )
    scope = {}
    exec(code, scope)

    if backend == "python":
        actual = [scope["formula"](value) for value in values]
    else:
        actual = scope["formula"](np.asarray(values))

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_numpy_piecewise_works_with_array_inputs():
    with pytest.warns(PiecewiseEvaluationWarning, match="eager elementwise selection"):
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


@pytest.mark.parametrize(
    ("backend", "imports", "annotation"),
    [
        ("torch", "import torch", "x: torch.Tensor"),
        ("jax", "import jax.numpy as jnp", "x: jax.Array"),
    ],
)
def test_framework_backend_source_and_annotations(backend, imports, annotation):
    code = transpile_latex(r"\sin(x) + x^2", backend=backend)
    assert imports in code
    assert annotation in code
    if backend == "jax":
        assert "import numpy as np" not in code
    compile(code, f"<{backend} generated source>", "exec")


def test_backend_rejects_unknown_and_conflicting_options():
    with pytest.raises(LaTeXTranspilerError, match="Unknown backend"):
        transpile_latex("x", backend="tensorflow")
    with pytest.raises(LaTeXTranspilerError, match="conflicts"):
        transpile_latex("x", backend="torch", use_numpy=True)


def test_torch_backend_matches_scalar_and_array_values_and_gradients():
    torch = pytest.importorskip("torch")
    code = transpile_latex(r"\sin(x)^2 + x", backend="torch", type_hints=False)
    scope = {}
    exec(code, scope)

    scalar = torch.tensor(0.4, requires_grad=True)
    scalar_result = scope["formula"](scalar)
    scalar_result.backward()
    assert math.isclose(scalar_result.item(), math.sin(0.4) ** 2 + 0.4, rel_tol=1e-6)
    assert math.isclose(scalar.grad.item(), math.sin(0.8) + 1, rel_tol=1e-6)

    values = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    array_result = scope["formula"](values)
    array_result.sum().backward()
    expected = torch.sin(values.detach()) ** 2 + values.detach()
    assert torch.allclose(array_result, expected)
    assert torch.allclose(values.grad, torch.sin(2 * values.detach()) + 1)


def test_jax_backend_matches_scalar_and_array_values_and_gradients():
    jax = pytest.importorskip("jax")
    jax_numpy = pytest.importorskip("jax.numpy")
    code = transpile_latex(r"\sin(x)^2 + x", backend="jax", type_hints=False)
    scope = {}
    exec(code, scope)
    formula = scope["formula"]

    scalar_result = formula(jax_numpy.asarray(0.4))
    scalar_gradient = jax.grad(formula)(jax_numpy.asarray(0.4))
    assert math.isclose(float(scalar_result), math.sin(0.4) ** 2 + 0.4, rel_tol=1e-6)
    assert math.isclose(float(scalar_gradient), math.sin(0.8) + 1, rel_tol=1e-6)

    values = jax_numpy.asarray([0.0, 0.5, 1.0])
    array_result = formula(values)
    expected = jax_numpy.sin(values) ** 2 + values
    assert jax_numpy.allclose(array_result, expected)


@pytest.mark.parametrize("backend", ["python", "numpy", "torch", "jax"])
def test_multivariable_scientific_expression_matches_values_and_gradients(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        jax_numpy = pytest.importorskip("jax.numpy")

    code = transpile_latex(r"\frac{\sin(x)}{x} + \sqrt{y}", backend=backend, type_hints=False)
    scope = {}
    exec(code, scope)
    formula = scope["formula"]

    x_values = [0.2, 0.7, 1.4]
    y_values = [0.25, 1.0, 2.25]
    expected = [
        math.sin(x) / x + math.sqrt(y)
        for x, y in zip(x_values, y_values)
    ]

    if backend == "python":
        actual = [formula(x, y) for x, y in zip(x_values, y_values)]
    elif backend == "numpy":
        actual = formula(np.array(x_values), np.array(y_values))
    elif backend == "torch":
        x = torch.tensor(x_values, requires_grad=True)
        y = torch.tensor(y_values, requires_grad=True)
        actual = formula(x, y)
        actual.sum().backward()
        expected_x_gradient = [
            (x * math.cos(x) - math.sin(x)) / x**2
            for x in x_values
        ]
        expected_y_gradient = [1 / (2 * math.sqrt(y)) for y in y_values]
        assert torch.allclose(x.grad, torch.tensor(expected_x_gradient), rtol=1e-5, atol=1e-6)
        assert torch.allclose(y.grad, torch.tensor(expected_y_gradient), rtol=1e-5, atol=1e-6)
    else:
        x = jax_numpy.asarray(x_values)
        y = jax_numpy.asarray(y_values)
        actual = formula(x, y)
        x_gradient, y_gradient = jax.grad(
            lambda xs, ys: jax_numpy.sum(formula(xs, ys)),
            argnums=(0, 1),
        )(x, y)
        expected_x_gradient = jax_numpy.asarray([
            (value * math.cos(value) - math.sin(value)) / value**2
            for value in x_values
        ])
        expected_y_gradient = jax_numpy.asarray([
            1 / (2 * math.sqrt(value))
            for value in y_values
        ])
        assert jax_numpy.allclose(x_gradient, expected_x_gradient, rtol=1e-5, atol=1e-6)
        assert jax_numpy.allclose(y_gradient, expected_y_gradient, rtol=1e-5, atol=1e-6)

    if backend == "torch":
        assert torch.allclose(actual.detach(), torch.tensor(expected), rtol=1e-6, atol=1e-7)
    elif backend == "jax":
        assert jax_numpy.allclose(actual, jax_numpy.asarray(expected), rtol=1e-6, atol=1e-7)
    else:
        assert np.allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_framework_matrix_outputs_preserve_gradients(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
    else:
        jax = pytest.importorskip("jax")
        jax_numpy = pytest.importorskip("jax.numpy")

    latex = r"\begin{pmatrix} x & x^2 \\ 0 & 1 \end{pmatrix}"
    code = transpile_latex(latex, backend=backend, type_hints=False)
    scope = {}
    exec(code, scope)

    if backend == "torch":
        value = torch.tensor(0.5, requires_grad=True)
        result = scope["formula"](value)
        result.sum().backward()
        assert torch.allclose(result, torch.tensor([[0.5, 0.25], [0.0, 1.0]]))
        assert math.isclose(value.grad.item(), 2.0, rel_tol=1e-6)
    else:
        value = jax_numpy.asarray(0.5)
        result = scope["formula"](value)
        gradient = jax.grad(lambda item: scope["formula"](item).sum())(value)
        assert jax_numpy.allclose(result, jax_numpy.asarray([[0.5, 0.25], [0.0, 1.0]]))
        assert math.isclose(float(gradient), 2.0, rel_tol=1e-6)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_framework_special_functions_support_values_and_gradients(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        values = torch.tensor([-1.5, -0.5, 0.5, 1.5, 5.0])
        gamma_gradient = torch.tensor(2.0, requires_grad=True)
    else:
        jax = pytest.importorskip("jax")
        jax_numpy = pytest.importorskip("jax.numpy")
        values = jax_numpy.asarray([-1.5, -0.5, 0.5, 1.5, 5.0])
        gamma_gradient = jax_numpy.asarray(2.0)

    for operator, expected in (
        ("erf", [math.erf(value) for value in (-1.5, -0.5, 0.5, 1.5, 5.0)]),
        ("erfc", [math.erfc(value) for value in (-1.5, -0.5, 0.5, 1.5, 5.0)]),
        ("gamma", [math.gamma(value) for value in (-1.5, -0.5, 0.5, 1.5, 5.0)]),
    ):
        code = transpile_latex(
            rf"\operatorname{{{operator}}}(x)",
            backend=backend,
            type_hints=False,
        )
        scope = {}
        exec(code, scope)
        result = scope["formula"](values)
        if backend == "torch":
            assert torch.allclose(result, torch.tensor(expected), rtol=1e-5, atol=1e-6)
        else:
            assert jax_numpy.allclose(result, jax_numpy.asarray(expected), rtol=1e-5, atol=1e-6)

    gamma_code = transpile_latex(r"\operatorname{gamma}(x)", backend=backend, type_hints=False)
    scope = {}
    exec(gamma_code, scope)
    if backend == "torch":
        scope["formula"](gamma_gradient).backward()
        expected_gradient = math.gamma(2.0) * float(torch.digamma(torch.tensor(2.0)))
        assert math.isclose(gamma_gradient.grad.item(), expected_gradient, rel_tol=1e-5)
    else:
        gradient = jax.grad(scope["formula"])(gamma_gradient)
        expected_gradient = math.gamma(2.0) * float(jax.scipy.special.digamma(2.0))
        assert math.isclose(float(gradient), expected_gradient, rel_tol=1e-5)


@pytest.mark.parametrize("backend", ["python", "numpy", "torch", "jax"])
def test_sigmoid_and_softplus_are_stable_for_extreme_inputs(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        values = torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
        expected_sigmoid = torch.tensor([0.0, 0.5, 1.0])
        expected_softplus = torch.tensor([0.0, math.log(2), 1000.0])
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        jax_numpy = pytest.importorskip("jax.numpy")
        values = jax_numpy.asarray([-1000.0, 0.0, 1000.0])
        expected_sigmoid = jax_numpy.asarray([0.0, 0.5, 1.0])
        expected_softplus = jax_numpy.asarray([0.0, math.log(2), 1000.0])
    elif backend == "numpy":
        values = np.array([-1000.0, 0.0, 1000.0])
        expected_sigmoid = np.array([0.0, 0.5, 1.0])
        expected_softplus = np.array([0.0, math.log(2), 1000.0])
    else:
        values = None
        expected_sigmoid = [0.0, 0.5, 1.0]
        expected_softplus = [0.0, math.log(2), 1000.0]

    for operator, expected in (
        ("sigmoid", expected_sigmoid),
        ("softplus", expected_softplus),
    ):
        code = transpile_latex(
            rf"\operatorname{{{operator}}}(x)",
            backend=backend,
            type_hints=False,
        )
        scope = {}
        exec(code, scope)
        if backend == "python":
            actual = [scope["formula"](value) for value in (-1000.0, 0.0, 1000.0)]
            assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        else:
            actual = scope["formula"](values)
            if backend == "torch":
                assert torch.isfinite(actual).all()
                assert torch.allclose(actual, expected)
            elif backend == "jax":
                assert jax_numpy.all(jax_numpy.isfinite(actual))
                assert jax_numpy.allclose(actual, expected)
            else:
                assert np.all(np.isfinite(actual))
                assert np.allclose(actual, expected)


def test_torch_ml_activations_keep_gradients_at_extreme_inputs():
    torch = pytest.importorskip("torch")
    for operator in ("sigmoid", "softplus", "swish"):
        code = transpile_latex(
            rf"\operatorname{{{operator}}}(x)",
            backend="torch",
            type_hints=False,
        )
        scope = {}
        exec(code, scope)
        values = torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
        result = scope["formula"](values)
        result.sum().backward()
        assert torch.isfinite(result).all()
        assert torch.isfinite(values.grad).all()
        expected = [0.0, 0.25, 0.0] if operator == "sigmoid" else [0.0, 0.5, 1.0]
        assert torch.allclose(values.grad, torch.tensor(expected), atol=1e-6)


def test_jax_ml_activations_keep_gradients_at_extreme_inputs():
    jax = pytest.importorskip("jax")
    jax_numpy = pytest.importorskip("jax.numpy")
    for operator in ("sigmoid", "softplus", "swish"):
        code = transpile_latex(
            rf"\operatorname{{{operator}}}(x)",
            backend="jax",
            type_hints=False,
        )
        scope = {}
        exec(code, scope)
        formula = scope["formula"]
        values = jax_numpy.asarray([-1000.0, 0.0, 1000.0])
        result = formula(values)
        gradients = jax.vmap(jax.grad(formula))(values)
        assert jax_numpy.all(jax_numpy.isfinite(result))
        assert jax_numpy.all(jax_numpy.isfinite(gradients))
        expected = [0.0, 0.25, 0.0] if operator == "sigmoid" else [0.0, 0.5, 1.0]
        assert jax_numpy.allclose(gradients, jax_numpy.asarray(expected), atol=1e-6)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_framework_ml_operators_support_scalar_array_and_gradient_workflows(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        array = torch.tensor
        assert_close = torch.allclose
    else:
        jax = pytest.importorskip("jax")
        jax_numpy = pytest.importorskip("jax.numpy")
        array = jax_numpy.asarray
        assert_close = jax_numpy.allclose

    cases = {
        "sigmoid": ([-2.0, 0.5, 2.0], lambda x: 1 / (1 + np.exp(-x)),
                    lambda x: (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))),
        "relu": ([-2.0, 0.5, 2.0], lambda x: np.maximum(x, 0),
                 lambda x: (x > 0).astype(float)),
        "softplus": ([-2.0, 0.5, 2.0], lambda x: np.logaddexp(0, x),
                     lambda x: 1 / (1 + np.exp(-x))),
        "logit": ([0.2, 0.5, 0.8], lambda x: np.log(x / (1 - x)),
                  lambda x: 1 / x + 1 / (1 - x)),
        "softsign": ([-2.0, 0.5, 2.0], lambda x: x / (1 + np.abs(x)),
                     lambda x: 1 / (1 + np.abs(x)) ** 2),
        "swish": ([-2.0, 0.5, 2.0], lambda x: x / (1 + np.exp(-x)),
                  lambda x: (1 / (1 + np.exp(-x))) + x * (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))),
    }

    for operator, (raw_values, expected_fn, gradient_fn) in cases.items():
        values = array(raw_values)
        expected = array(expected_fn(np.asarray(raw_values)), dtype=values.dtype)
        expected_gradient = array(gradient_fn(np.asarray(raw_values)), dtype=values.dtype)
        code = transpile_latex(
            rf"\operatorname{{{operator}}}(x)",
            backend=backend,
            type_hints=False,
        )
        scope = {}
        exec(code, scope)
        formula = scope["formula"]
        assert assert_close(formula(values), expected, rtol=1e-5, atol=1e-6)
        assert assert_close(formula(values[0]), expected[0], rtol=1e-5, atol=1e-6)

        if backend == "torch":
            differentiable_values = torch.tensor(raw_values, requires_grad=True)
            formula(differentiable_values).sum().backward()
            assert assert_close(
                differentiable_values.grad,
                expected_gradient,
                rtol=1e-5,
                atol=1e-6,
            )
        else:
            gradients = jax.vmap(jax.grad(formula))(values)
            assert assert_close(gradients, expected_gradient, rtol=1e-5, atol=1e-6)


def test_torch_backend_preserves_tensor_dtype_device_and_batch_shape():
    torch = pytest.importorskip("torch")
    code = transpile_latex(
        r"\operatorname{relu}(x) + x^2",
        backend="torch",
        type_hints=False,
    )
    scope = {}
    exec(code, scope)
    values = torch.tensor([[-2.0, 0.5], [1.0, 2.0]], dtype=torch.float64)
    result = scope["formula"](values)
    expected = torch.relu(values) + values**2
    assert result.dtype == values.dtype
    assert result.device == values.device
    assert result.shape == values.shape
    assert torch.allclose(result, expected)


def test_jax_backend_is_jittable_and_supports_vmap():
    jax = pytest.importorskip("jax")
    jax_numpy = pytest.importorskip("jax.numpy")
    code = transpile_latex(
        r"\operatorname{softplus}(x) + x^2",
        backend="jax",
        type_hints=False,
    )
    scope = {}
    exec(code, scope)
    formula = scope["formula"]
    values = jax_numpy.asarray([-2.0, 0.5, 2.0])
    expected = jax.nn.softplus(values) + values**2
    assert jax_numpy.allclose(jax.jit(formula)(values), expected)
    mapped = jax.vmap(formula)(values)
    assert jax_numpy.allclose(mapped, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_array_piecewise_generation_warns_about_eager_branch_evaluation(backend):
    with pytest.warns(PiecewiseEvaluationWarning, match="all branch expressions may be evaluated"):
        transpile_latex(
            r"\begin{cases} \log(x) & x > 0 \\ 0 & x \le 0 \end{cases}",
            backend=backend,
            type_hints=False,
        )


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


def test_diagnostics_suggest_close_command_and_function_names():
    with pytest.raises(UnsupportedLaTeXFeatureError, match=r"Did you mean '\\\\sin'") as error:
        transpile_latex("x +\n  \\sinn(y)")
    assert error.value.line == 2
    assert error.value.column == 3
    assert "\\sinn(y)" in str(error.value)
    with pytest.raises(UnsupportedLaTeXFeatureError, match="sigmoid"):
        transpile_latex(r"\operatorname{sigmiod}(x)")


def test_syntax_error_reports_unmatched_delimiter_location():
    expression = r"\frac{x}{2"
    with pytest.raises(InvalidLaTeXSyntaxError, match="Location: line 1, column 9") as error:
        transpile_latex(expression)

    assert error.value.line == 1
    assert error.value.column == 9
    assert "  \\frac{x}{2\n          ^" in str(error.value)
    assert "curly braces are balanced" in str(error.value)


def test_syntax_error_reports_multiline_environment_location():
    expression = "\\begin{pmatrix}\n  x & y\n\\end{bmatrix}"
    with pytest.raises(InvalidLaTeXSyntaxError, match="Location: line 1, column 1") as error:
        transpile_latex(expression)

    assert error.value.line == 1
    assert error.value.column == 1


@pytest.mark.parametrize(
    ("expression", "line", "column", "hint"),
    [
        ("x +", 1, 4, "ends with an operator"),
        ("x + * y", 1, 5, "consecutive operators"),
    ],
)
def test_strict_parser_reports_common_expression_typos(expression, line, column, hint):
    with pytest.raises(InvalidLaTeXSyntaxError) as error:
        transpile_latex(expression)

    assert (error.value.line, error.value.column) == (line, column)
    assert hint in str(error.value)


def test_safe_fallback_rejects_non_math_python_syntax():
    with pytest.raises(InvalidLaTeXSyntaxError):
        transpile_latex(r"\mathrm{erf}(__import__(x))")


def test_matrix_rows_must_have_matching_dimensions():
    with pytest.raises(InvalidLaTeXSyntaxError, match=r"Location: line 1, column 26") as error:
        transpile_latex(r"\begin{pmatrix} x & y \\ z \end{pmatrix}")
    assert "same number of entries" in str(error.value)


def test_malformed_matrix_entry_reports_nested_source_location():
    expression = "\\begin{pmatrix}\n x & \\sin(\n 0 & 1\n\\end{pmatrix}"

    with pytest.raises(InvalidLaTeXSyntaxError, match=r"Location: line 2, column 10") as error:
        transpile_latex(expression)

    assert error.value.line == 2
    assert error.value.column == 10
    assert "\\sin(" in str(error.value)


def test_malformed_piecewise_expression_reports_nested_source_location():
    expression = "\\begin{cases}\n x + & x > 0 \\\\ 0 & otherwise\n\\end{cases}"

    with pytest.raises(InvalidLaTeXSyntaxError, match=r"Location: line 2, column 5") as error:
        transpile_latex(expression)

    assert error.value.line == 2
    assert error.value.column == 5


def test_piecewise_row_without_separator_points_to_offending_row():
    expression = "\\begin{cases}\n x & x > 0 \\\\ malformed\n\\end{cases}"

    with pytest.raises(InvalidLaTeXSyntaxError, match=r"Location: line 2, column 15") as error:
        transpile_latex(expression)

    assert error.value.line == 2
    assert error.value.column == 15
    assert "Each cases row must separate" in str(error.value)


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


def test_cli_inspect_prints_json_report(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["latex2code", r"\pi + \theta", "--inspect", "--no-types"],
    )

    cli_main()

    report = json.loads(capsys.readouterr().out)
    assert report["parsed_expression"] == "theta + pi"
    assert report["variables"] == ["theta"]
    assert "math.pi" in report["generated_code"]
    assert report["expression_tree"]["type"] == "Add"
    assert report["warnings"] == []


def test_cli_maps_variables_to_python_argument_names(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["latex2code", r"\theta + x", "--map-variable", "theta=angle", "--no-types"],
    )

    cli_main()

    output = capsys.readouterr().out
    assert "def formula(angle, x):" in output
    assert "return angle + x" in output


def test_cli_interactive_generates_multiple_expressions(monkeypatch, capsys):
    expressions = iter([r"\pi x", r"\sin(y)", "quit"])
    monkeypatch.setattr("sys.argv", ["latex2code", "--interactive", "--no-types"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(expressions))

    cli_main()

    output = capsys.readouterr().out
    assert "latex2code interactive" in output
    assert "return math.pi*x" in output
    assert "return math.sin(y)" in output


def test_cli_interactive_accepts_multiline_expression(monkeypatch, capsys):
    expressions = iter([":begin", r"\frac{1}{", r"x + y}", ":end", "quit"])
    monkeypatch.setattr("sys.argv", ["latex2code", "--interactive", "--no-types"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(expressions))

    cli_main()

    assert "return 1/(x + y)" in capsys.readouterr().out


def test_cli_interactive_reports_bad_expression_and_continues(monkeypatch, capsys):
    expressions = iter(["x +", "y", "quit"])
    monkeypatch.setattr("sys.argv", ["latex2code", "--interactive", "--no-types"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(expressions))

    cli_main()

    captured = capsys.readouterr()
    assert "ends with an operator" in captured.err
    assert "return y" in captured.out


def test_cli_inspect_rejects_output_file_option(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["latex2code", "x", "--inspect", "--output", "generated.py"],
    )

    with pytest.raises(SystemExit) as error:
        cli_main()

    assert error.value.code == 2


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


def test_cli_creates_parent_directories_for_output(monkeypatch, tmp_path):
    output_file = tmp_path / "missing" / "generated.py"
    monkeypatch.setattr(
        "sys.argv",
        ["latex2code", "x", "--output", str(output_file)],
    )

    cli_main()

    assert output_file.read_text(encoding="utf-8") == "def formula(x: float) -> float:\n    return x\n"


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


def test_nth_roots_and_inverse_trig_are_supported():
    root_code = transpile_latex(r"\sqrt[3]{x}", type_hints=False)
    root_scope = {}
    exec(root_code, root_scope)
    assert math.isclose(root_scope["formula"](8), 2.0)

    reciprocal_code = transpile_latex(r"\csc(x)", type_hints=False)
    reciprocal_scope = {}
    exec(reciprocal_code, reciprocal_scope)
    assert math.isclose(reciprocal_scope["formula"](math.pi / 2), 1.0)

    acot_code = transpile_latex(r"\arccot(x)", type_hints=False)
    acot_scope = {}
    exec(acot_code, acot_scope)
    assert math.isclose(acot_scope["formula"](1.0), math.pi / 4)


def test_array_environment_supports_matrix_style_layouts():
    code = transpile_latex(r"\begin{array}{cc} a & b \\ c & d \end{array}", use_numpy=True, type_hints=False)
    scope = {"np": np}
    exec(code, scope)
    result = scope["formula"](1, 2, 3, 4)
    expected = np.array([[1, 2], [3, 4]])
    assert np.array_equal(result, expected)


def test_cli_supports_version_flag(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["latex2code", "--version"])
    with pytest.raises(SystemExit) as error:
        cli_main()
    assert error.value.code == 0
    assert f"latex2code {__version__}" in capsys.readouterr().out


def test_infinity_constant_and_hypot_operator_are_supported():
    infinity_code = transpile_latex(r"\infty", type_hints=False)
    infinity_scope = {}
    exec(infinity_code, infinity_scope)
    assert infinity_scope["formula"]() == math.inf

    hypot_code = transpile_latex(r"\operatorname{hypot}(x, y)", type_hints=False)
    hypot_scope = {}
    exec(hypot_code, hypot_scope)
    assert math.isclose(hypot_scope["formula"](3, 4), 5.0)


@pytest.mark.parametrize(
    ("operator", "values", "expected"),
    [
        ("sigmoid", (np.array([-2.0, 0.0, 2.0]),), 1 / (1 + np.exp(-np.array([-2.0, 0.0, 2.0])))),
        ("relu", (np.array([-2.0, 0.0, 2.0]),), np.array([0.0, 0.0, 2.0])),
        ("softplus", (np.array([-2.0, 0.0, 2.0]),), np.log1p(np.exp(np.array([-2.0, 0.0, 2.0])))),
        ("logit", (np.array([0.2, 0.5, 0.8]),), np.log(np.array([0.2, 0.5, 0.8]) / (1 - np.array([0.2, 0.5, 0.8])))),
        ("softsign", (np.array([-2.0, 0.0, 2.0]),), np.array([-2.0, 0.0, 2.0]) / (1 + np.abs(np.array([-2.0, 0.0, 2.0])))),
        ("swish", (np.array([-2.0, 0.0, 2.0]),), np.array([-2.0, 0.0, 2.0]) / (1 + np.exp(-np.array([-2.0, 0.0, 2.0])))),
    ],
)
def test_machine_learning_operators_support_numpy_arrays(operator, values, expected):
    code = transpile_latex(
        rf"\operatorname{{{operator}}}(x)",
        use_numpy=True,
        type_hints=False,
    )
    scope = {}
    exec(code, scope)
    assert np.allclose(scope["formula"](*values), expected)


def test_machine_learning_operators_support_scalar_inputs():
    for operator, value, expected in (
        ("sigmoid", 0.0, 0.5),
        ("relu", -3.0, 0.0),
        ("softplus", 0.0, math.log(2)),
        ("logit", 0.5, 0.0),
        ("softsign", -2.0, -2 / 3),
        ("swish", 0.0, 0.0),
    ):
        code = transpile_latex(
            rf"\operatorname{{{operator}}}(x)",
            type_hints=False,
        )
        scope = {}
        exec(code, scope)
        assert math.isclose(scope["formula"](value), expected)


@pytest.mark.parametrize("backend", ["python", "numpy", "torch", "jax"])
def test_beta_function_is_stable_and_differentiable_where_supported(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
    elif backend == "jax":
        jax = pytest.importorskip("jax")
        jax_numpy = pytest.importorskip("jax.numpy")

    code = transpile_latex(
        r"\operatorname{beta}(x, y)",
        backend=backend,
        type_hints=False,
    )
    scope = {}
    exec(code, scope)

    expected = [
        (
            math.copysign(1.0, math.sin(math.pi * x)) if x < 0 else 1.0
        ) * (
            math.copysign(1.0, math.sin(math.pi * y)) if y < 0 else 1.0
        ) / (
            math.copysign(1.0, math.sin(math.pi * (x + y))) if x + y < 0 else 1.0
        ) * math.exp(math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y))
        for x, y in ((-0.5, 2.0), (2.0, 3.0), (100.0, 100.0))
    ]

    if backend == "python":
        actual = [
            scope["formula"](-0.5, 2.0),
            scope["formula"](2.0, 3.0),
            scope["formula"](100.0, 100.0),
        ]
        assert np.allclose(actual, expected, rtol=1e-12, atol=0)
    elif backend == "numpy":
        actual = scope["formula"](
            np.array([-0.5, 2.0, 100.0]),
            np.array([2.0, 3.0, 100.0]),
        )
        assert np.all(np.isfinite(actual))
        assert np.allclose(actual, expected, rtol=1e-12, atol=0)
    elif backend == "torch":
        values = torch.tensor([-0.5, 2.0, 100.0], dtype=torch.float64)
        second = torch.tensor([2.0, 3.0, 100.0], dtype=torch.float64)
        actual = scope["formula"](values, second)
        assert torch.isfinite(actual).all()
        assert np.allclose(actual.detach().numpy(), expected, rtol=1e-12, atol=0)

        x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        scope["formula"](x, torch.tensor(3.0, dtype=torch.float64)).backward()
        expected_gradient = expected[1] * (
            float(torch.digamma(torch.tensor(2.0, dtype=torch.float64)))
            - float(torch.digamma(torch.tensor(5.0, dtype=torch.float64)))
        )
        assert math.isclose(x.grad.item(), expected_gradient, rel_tol=1e-10)
    else:
        values = jax_numpy.asarray([-0.5, 2.0, 100.0])
        second = jax_numpy.asarray([2.0, 3.0, 100.0])
        actual = scope["formula"](values, second)
        assert jax_numpy.all(jax_numpy.isfinite(actual))
        assert jax_numpy.allclose(actual, jax_numpy.asarray(expected), rtol=1e-5, atol=1e-7)

        gradient = jax.grad(lambda x: scope["formula"](x, 3.0))(jax_numpy.asarray(2.0))
        expected_gradient = expected[1] * (
            float(jax.scipy.special.digamma(2.0))
            - float(jax.scipy.special.digamma(5.0))
        )
        assert math.isclose(float(gradient), expected_gradient, rel_tol=1e-5)


def test_aligned_equation_environments_are_matrix_like():
    code = transpile_latex(r"\begin{aligned} x &= y \\ z &= w \end{aligned}", use_numpy=True, type_hints=False)
    scope = {"np": np}
    exec(code, scope)
    result = scope["formula"](1, 2, 3, 4)
    expected = np.array([[1, 2], [3, 4]])
    assert np.array_equal(result, expected)


def test_invalid_syntax():
    with pytest.raises(InvalidLaTeXSyntaxError, match="curly braces are balanced"):
        transpile_latex(r"\frac{x}{2")