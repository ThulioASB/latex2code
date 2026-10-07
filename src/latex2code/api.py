import importlib.util
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import sympy as sp

from .backends import BACKENDS, Backend
from .errors import (
    CodeGenerationError,
    InvalidLaTeXSyntaxError,
    LaTeXTranspilerError,
    PiecewiseEvaluationWarning,
    _validate_python_identifier,
)
from .normalize import _normalize_latex_string
from .parsing import (
    _LATEX_PARSE_ERRORS,
    _check_for_undefined_commands,
    _ordered_free_symbols,
    _parse_cases_environment,
    _parse_expression,
    _parse_matrix_environment,
    _parse_operatorname_expression,
)
from .printers import _expression_tree, _FrameworkPrinter, _PythonPrinter


@dataclass(frozen=True)
class TranspilationInfo:
    """Parsed expression and generated source returned by :func:`inspect_latex`."""

    input_expression: str
    backend: str
    parsed_expression: str
    expression_tree: dict[str, object]
    variables: tuple[str, ...]
    generated_code: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CompiledFormula:
    """Callable generated formula together with its source and inspection details."""

    info: TranspilationInfo
    function: Callable[..., object]

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.function(*args, **kwargs)

    @property
    def source(self) -> str:
        return self.info.generated_code


def _transpile_latex(
    latex_str: str,
    *,
    function_name: str,
    type_hints: bool,
    use_numpy: bool | None,
    backend: Backend | None,
    variable_map: Mapping[str, str] | None,
    args: Sequence[str] | None,
    constants: Mapping[str, object] | None,
    warning_stacklevel: int,
) -> TranspilationInfo:
    """Build generated source and inspection details from a LaTeX expression."""
    if not latex_str or not latex_str.strip():
        raise LaTeXTranspilerError("LaTeX expression string cannot be empty.")
    _validate_python_identifier(function_name, "Function name")
    if use_numpy is not None:
        warnings.warn(
            "use_numpy is deprecated; use backend='numpy' instead.",
            DeprecationWarning,
            stacklevel=warning_stacklevel,
        )
    if backend is None:
        backend = "numpy" if use_numpy else "python"
    elif not isinstance(backend, str) or backend not in {
        "python",
        "numpy",
        "torch",
        "jax",
    }:
        raise LaTeXTranspilerError(
            f"Unknown backend {backend!r}; choose 'python', 'numpy', 'torch', or 'jax'."
        )
    if use_numpy and backend != "numpy":
        raise LaTeXTranspilerError(
            "use_numpy=True conflicts with an explicit backend; use backend='numpy' instead."
        )
    backend_spec = BACKENDS[backend]
    normalized_latex = _normalize_latex_string(latex_str)

    # 1. Parse matrix, piecewise, or general expression
    try:
        expr = _parse_matrix_environment(normalized_latex)
        is_matrix = expr is not None

        if not is_matrix:
            expr = _parse_cases_environment(normalized_latex)
            if expr is not None:
                is_matrix = False

        if expr is None:
            _check_for_undefined_commands(sp.Symbol("temp"), normalized_latex)
            if r"\operatorname" in normalized_latex:
                expr = _parse_operatorname_expression(normalized_latex)
            else:
                expr = _parse_expression(normalized_latex)
            _check_for_undefined_commands(expr, normalized_latex)

            # Evaluate derivatives, limits, or integrals symbolically if present
            if (
                expr.has(sp.Derivative)
                or expr.has(sp.Integral)
                or expr.has(sp.Limit)
                or expr.has(sp.Product)
            ):
                expr = expr.doit()
    except _LATEX_PARSE_ERRORS as exc:
        raise InvalidLaTeXSyntaxError(latex_str, original_error=exc) from exc

    constant_values: Mapping[str, object]
    if constants is None:
        constant_values = {"pi": sp.pi, "e": sp.E}
    else:
        if not isinstance(constants, Mapping):
            raise LaTeXTranspilerError(
                "constants must map symbol names to numeric values."
            )
        if not all(
            isinstance(name, str) and isinstance(value, (int, float, complex, sp.Expr))
            for name, value in constants.items()
        ):
            raise LaTeXTranspilerError(
                "constant names must be strings and values must be numbers or SymPy expressions."
            )
        unused_constants = (
            set(constants) - {str(symbol) for symbol in expr.free_symbols} - {"pi"}
        )
        if unused_constants:
            raise LaTeXTranspilerError(
                "constants contains symbols not present in the expression: "
                f"{', '.join(sorted(unused_constants))}."
            )
        constant_values = constants
    expr = expr.subs(
        {sp.Symbol(name): sp.sympify(value) for name, value in constant_values.items()}
    )

    # 2. Extract free variables
    variables = _ordered_free_symbols(expr, normalized_latex)
    if args is not None:
        if (
            isinstance(args, (str, bytes))
            or not isinstance(args, Sequence)
            or not all(isinstance(argument, str) for argument in args)
        ):
            raise LaTeXTranspilerError("args must be a sequence of symbol names.")
        if len(set(args)) != len(args) or set(args) != set(variables):
            raise LaTeXTranspilerError(
                "args must contain every expression variable exactly once."
            )
        variables = list(args)
    if variable_map is not None:
        if not isinstance(variable_map, Mapping):
            raise LaTeXTranspilerError(
                "variable_map must map symbol names to Python identifiers."
            )
        if not all(
            isinstance(source, str) and isinstance(target, str)
            for source, target in variable_map.items()
        ):
            raise LaTeXTranspilerError("variable_map keys and values must be strings.")
        unknown_symbols = set(variable_map) - set(variables)
        if unknown_symbols:
            raise LaTeXTranspilerError(
                "variable_map contains symbols not present in the expression: "
                f"{', '.join(sorted(unknown_symbols))}."
            )
        mapped_variables = [
            variable_map.get(variable, variable) for variable in variables
        ]
        for variable in mapped_variables:
            _validate_python_identifier(variable, "Mapped argument name")
        if len(set(mapped_variables)) != len(mapped_variables):
            raise LaTeXTranspilerError(
                "variable_map must not map multiple symbols to the same Python identifier."
            )
        expr = expr.xreplace(
            {
                sp.Symbol(source): sp.Symbol(target)
                for source, target in variable_map.items()
            }
        )
        variables = mapped_variables
    transpilation_warnings = []
    for variable in variables:
        _validate_python_identifier(variable, "Generated argument name")
    if backend in {"numpy", "torch", "jax"} and expr.has(sp.Piecewise):
        warning_message = (
            "Array-backend piecewise expressions use eager elementwise selection; "
            "all branch expressions may be evaluated, including branches not selected "
            "for an element. Keep every branch valid over the full input domain."
        )
        transpilation_warnings.append(warning_message)
        warnings.warn(
            warning_message,
            PiecewiseEvaluationWarning,
            stacklevel=warning_stacklevel,
        )

    # 3. Build argument signature
    argument_type = backend_spec.argument_type
    return_type = (
        backend_spec.matrix_return_type
        if is_matrix
        else backend_spec.scalar_return_type
    )
    if type_hints:
        args_str = ", ".join([f"{var}: {argument_type}" for var in variables])
        return_hint = f" -> {return_type}"
    else:
        args_str = ", ".join(variables)
        return_hint = ""

    # 4. Generate Python code representation
    printer: _PythonPrinter | _FrameworkPrinter | None = None
    try:
        if is_matrix and backend in {"torch", "jax"}:
            printer = _FrameworkPrinter(backend)
            matrix_rows = [
                "[" + ", ".join(printer.doprint(value) for value in row) + "]"
                for row in expr.tolist()
            ]
            if backend == "torch" and variables:
                like = variables[0]
                rows_with_tensors = [
                    "["
                    + ", ".join(
                        f"({value} if torch.is_tensor({value}) else "
                        f"torch.as_tensor({value}, dtype={like}.dtype, device={like}.device))"
                        for value in row
                    )
                    + "]"
                    for row in expr.tolist()
                ]
                python_expr_code = (
                    "torch.stack([torch.stack(row) for row in ["
                    + ", ".join(rows_with_tensors)
                    + "]])"
                )
            elif backend == "torch":
                python_expr_code = f"torch.tensor([{', '.join(matrix_rows)}])"
            else:
                python_expr_code = f"jnp.array([{', '.join(matrix_rows)}])"
        elif is_matrix:
            matrix_list = expr.tolist()
            module = backend_spec.module_alias or "np"
            python_expr_code = f"{module}.array({matrix_list})"
        elif backend == "numpy":
            printer = _FrameworkPrinter(backend)
            python_expr_code = printer.doprint(expr)
        elif backend in {"torch", "jax"}:
            printer = _FrameworkPrinter(backend)
            python_expr_code = printer.doprint(expr)
        else:
            printer = _PythonPrinter()
            python_expr_code = printer.doprint(expr)
        printer_modules = set(printer.module_imports) if printer is not None else set()
        if backend in {"torch", "jax"} and "math" in printer_modules:
            raise CodeGenerationError(
                f"The {backend} backend has no differentiable native implementation for "
                f"part of this expression: {expr}"
            )
    except Exception as exc:
        if isinstance(exc, CodeGenerationError):
            raise
        raise CodeGenerationError(
            f"Unable to generate Python code for parsed expression: {expr}"
        ) from exc

    # 5. Assemble required imports
    imports_list = list(backend_spec.imports)
    if backend == "python":
        if is_matrix:
            imports_list.append("import numpy as np")
        for module in ("builtins", "math", "functools"):
            if module in printer_modules:
                imports_list.append(f"import {module}")

    imports = "\n".join(imports_list)
    if imports:
        imports += "\n\n"

    # 6. Construct and return complete source code
    code = f"{imports}def {function_name}({args_str}){return_hint}:\n"
    code += f"    return {python_expr_code}\n"

    return TranspilationInfo(
        input_expression=latex_str,
        backend=backend,
        parsed_expression=str(expr),
        expression_tree=_expression_tree(expr),
        variables=tuple(variables),
        generated_code=code,
        warnings=tuple(transpilation_warnings),
    )


def transpile_latex(
    latex_str: str,
    *,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool | None = None,
    backend: Backend | None = None,
    variable_map: Mapping[str, str] | None = None,
    args: Sequence[str] | None = None,
    constants: Mapping[str, object] | None = None,
    _warning_stacklevel: int = 3,
) -> str:
    """Convert LaTeX into Python source targeting Python math, NumPy, PyTorch, or JAX."""
    return _transpile_latex(
        latex_str,
        function_name=function_name,
        type_hints=type_hints,
        use_numpy=use_numpy,
        backend=backend,
        variable_map=variable_map,
        args=args,
        constants=constants,
        warning_stacklevel=_warning_stacklevel,
    ).generated_code


def inspect_latex(
    latex_str: str,
    *,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool | None = None,
    backend: Backend | None = None,
    variable_map: Mapping[str, str] | None = None,
    args: Sequence[str] | None = None,
    constants: Mapping[str, object] | None = None,
    _warning_stacklevel: int = 3,
) -> TranspilationInfo:
    """Return the parsed expression, variables, warnings, and generated source."""
    return _transpile_latex(
        latex_str,
        function_name=function_name,
        type_hints=type_hints,
        use_numpy=use_numpy,
        backend=backend,
        variable_map=variable_map,
        args=args,
        constants=constants,
        warning_stacklevel=_warning_stacklevel,
    )


def compile_latex(
    latex_str: str,
    *,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool | None = None,
    backend: Backend | None = None,
    variable_map: Mapping[str, str] | None = None,
    args: Sequence[str] | None = None,
    constants: Mapping[str, object] | None = None,
) -> CompiledFormula:
    """Compile LaTeX into a callable and retain its generated source and metadata."""
    selected_backend = backend or ("numpy" if use_numpy else "python")
    if selected_backend == "torch" and importlib.util.find_spec("torch") is None:
        raise LaTeXTranspilerError(
            "The 'torch' backend requires the optional dependency 'torch'. "
            "Install it with `pip install latex2code[torch]`."
        )
    if selected_backend == "jax" and importlib.util.find_spec("jax") is None:
        raise LaTeXTranspilerError(
            "The 'jax' backend requires the optional dependency 'jax'. "
            "Install it with `pip install latex2code[jax]`."
        )
    info = inspect_latex(
        latex_str,
        function_name=function_name,
        type_hints=type_hints,
        use_numpy=use_numpy,
        backend=backend,
        variable_map=variable_map,
        args=args,
        constants=constants,
        _warning_stacklevel=4,
    )
    namespace: dict[str, object] = {}
    exec(info.generated_code, namespace)
    function = namespace[function_name]
    if not callable(function):
        raise CodeGenerationError(
            f"Generated object {function_name!r} is not callable."
        )
    return CompiledFormula(info=info, function=function)


def latex_to_code(
    latex_str: str,
    *,
    custom_symbol_map: dict[str, str] | None = None,
    function_name: str = "formula",
    type_hints: bool = True,
    use_numpy: bool | None = None,
    backend: Backend | None = None,
    variable_map: Mapping[str, str] | None = None,
    args: Sequence[str] | None = None,
    constants: Mapping[str, object] | None = None,
) -> str:
    """Convert LaTeX into executable Python code using the main transpile pipeline.

    This helper preserves the historical `custom_symbol_map` behavior for macro
    overrides while also accepting the same generation options as
    :func:`transpile_latex`. Values are remapped via the standard variable map
    after the LaTeX parser has interpreted the expression.
    """
    merged_variable_map = dict(variable_map) if variable_map is not None else {}

    if custom_symbol_map is not None:
        warnings.warn(
            "custom_symbol_map is deprecated; use variable_map instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for macro, python_var in custom_symbol_map.items():
            if not isinstance(macro, str) or not macro.startswith("\\"):
                raise LaTeXTranspilerError(
                    "custom_symbol_map keys must be LaTeX command strings like '\\lambda'."
                )
            symbol_name = macro.lstrip("\\")
            if not re.fullmatch(r"[A-Za-z]+", symbol_name):
                raise LaTeXTranspilerError(
                    "custom_symbol_map supports plain LaTeX symbol commands such as '\\lambda'; "
                    f"received {macro!r}."
                )
            merged_variable_map[symbol_name] = python_var

    return transpile_latex(
        latex_str,
        function_name=function_name,
        type_hints=type_hints,
        use_numpy=use_numpy,
        backend=backend,
        variable_map=merged_variable_map or None,
        args=args,
        constants=constants,
        _warning_stacklevel=4,
    )
