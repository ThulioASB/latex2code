import sympy as sp
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.pycode import PythonCodePrinter

from .backends import BACKENDS, Backend
from .parsing import _StableBeta, _StableSigmoid, _StableSoftplus, _StableSwish


class _PythonPrinter(PythonCodePrinter):
    def _require_math(self) -> None:
        self.module_imports["math"].add("*")

    def _print__StableSigmoid(self, expr: _StableSigmoid) -> str:
        self._require_math()
        value = self._print(expr.args[0])
        return f"(0.5 * (1 + math.tanh(({value}) / 2)))"

    def _print__StableSoftplus(self, expr: _StableSoftplus) -> str:
        self._require_math()
        value = self._print(expr.args[0])
        return f"(max(({value}), 0) + math.log1p(math.exp(-abs({value}))))"

    def _print__StableSwish(self, expr: _StableSwish) -> str:
        value = self._print(expr.args[0])
        sigmoid = self._print(_StableSigmoid(expr.args[0]))
        return f"(({value}) * {sigmoid})"

    def _print__StableBeta(self, expr: _StableBeta) -> str:
        self._require_math()
        first, second = (self._print(value) for value in expr.args)
        sign_first = (
            f"(1 if ({first}) > 0 else math.copysign(1, math.sin(math.pi * ({first}))))"
        )
        sign_second = f"(1 if ({second}) > 0 else math.copysign(1, math.sin(math.pi * ({second}))))"
        total = f"(({first}) + ({second}))"
        sign_total = (
            f"(1 if {total} > 0 else math.copysign(1, math.sin(math.pi * {total})))"
        )
        log_magnitude = (
            f"(math.lgamma({first}) + math.lgamma({second}) - math.lgamma({total}))"
        )
        return f"(({sign_first}) * ({sign_second}) / ({sign_total}) * math.exp({log_magnitude}))"

    def _print_Sum(self, expr: sp.Sum) -> str:
        self.module_imports["builtins"].add("sum")
        return super()._print_Sum(expr)


class _FrameworkPrinter(NumPyPrinter):
    def __init__(self, backend: Backend):
        self.backend = backend
        self.backend_spec = BACKENDS[backend]
        if self.backend_spec.module_alias is None:
            raise ValueError(f"Backend {backend!r} does not use an array printer.")
        self._module = self.backend_spec.module_alias
        user_functions = {
            name: f"{self._module}.{target}"
            for name, target in self.backend_spec.function_aliases.items()
        }
        user_functions.update(self.backend_spec.special_functions)
        super().__init__(
            {
                "user_functions": user_functions,
            }
        )

    def _print_Pi(self, expr: sp.Expr) -> str:
        return f"{self.backend_spec.module_alias}.pi"

    def _print_Exp1(self, expr: sp.Expr) -> str:
        return f"{self.backend_spec.module_alias}.e"

    def _print_Abs(self, expr: sp.Expr) -> str:
        value = self._print(expr.args[0])
        return f"{self.backend_spec.module_alias}.abs({value})"

    def _print_gamma(self, expr: sp.Expr) -> str:
        value = self._print(expr.args[0])
        if self.backend == "torch":
            return (
                f"torch.where(({value}) < 0, "
                f"torch.sign(torch.sin(torch.pi * ({value}))), 1) "
                f"* torch.exp(torch.lgamma({value}))"
            )
        if self.backend == "numpy":
            return f"np.vectorize(math.gamma)({value})"
        return f"jax.scipy.special.gamma({value})"

    def _print__StableSigmoid(self, expr: _StableSigmoid) -> str:
        value = self._print(expr.args[0])
        if self.backend == "numpy":
            return f"np.exp(-np.logaddexp(0, -({value})))"
        module = "torch" if self.backend == "torch" else "jax.nn"
        return f"{module}.sigmoid({value})"

    def _print__StableSoftplus(self, expr: _StableSoftplus) -> str:
        value = self._print(expr.args[0])
        if self.backend == "numpy":
            return f"np.logaddexp(0, {value})"
        module = "torch.nn.functional" if self.backend == "torch" else "jax.nn"
        return f"{module}.softplus({value})"

    def _print__StableSwish(self, expr: _StableSwish) -> str:
        value = self._print(expr.args[0])
        return f"({value} * {self._print(_StableSigmoid(expr.args[0]))})"

    def _print__StableBeta(self, expr: _StableBeta) -> str:
        first, second = (self._print(value) for value in expr.args)
        total = f"({first} + {second})"
        if self.backend == "numpy":
            log_magnitude = (
                f"np.vectorize(math.lgamma)({first})"
                f" + np.vectorize(math.lgamma)({second})"
                f" - np.vectorize(math.lgamma)({total})"
            )
            sign = (
                f"np.where({first} < 0, np.sign(np.sin(np.pi * {first})), 1)"
                f" * np.where({second} < 0, np.sign(np.sin(np.pi * {second})), 1)"
                f" / np.where({total} < 0, np.sign(np.sin(np.pi * {total})), 1)"
            )
            return f"({sign} * np.exp({log_magnitude}))"
        module = "torch" if self.backend == "torch" else "jnp"
        if self.backend == "torch":
            log_gamma = "torch.lgamma"

            def sign_gamma(value: str) -> str:
                return (
                    f"torch.where(({value}) < 0, "
                    f"torch.sign(torch.sin(torch.pi * ({value}))), 1)"
                )

        else:
            log_gamma = "jax.scipy.special.gammaln"

            def sign_gamma(value: str) -> str:
                return f"jax.scipy.special.gammasgn({value})"

        log_magnitude = (
            f"{log_gamma}({first}) + {log_gamma}({second}) - {log_gamma}({total})"
        )
        sign = f"{sign_gamma(first)} * {sign_gamma(second)} / {sign_gamma(total)}"
        return f"({sign} * {module}.exp({log_magnitude}))"

    def _print_Piecewise(self, expr: sp.Piecewise) -> str:
        module = self.backend_spec.module_alias
        assert module is not None
        default = "float('nan')" if self.backend == "torch" else f"{module}.nan"
        branches = list(expr.args)
        for value, condition in reversed(branches):
            printed_value = self._print(value)
            if condition is sp.true:
                default = printed_value
                continue
            printed_condition = self._print(condition)
            default = f"{module}.where({printed_condition}, {printed_value}, {default})"
        return default

    def _print_Max(self, expr: sp.Max) -> str:
        return self._print_extreme(expr, "maximum")

    def _print_Min(self, expr: sp.Min) -> str:
        return self._print_extreme(expr, "minimum")

    def _print_extreme(self, expr: sp.Expr, operation: str) -> str:
        module = self.backend_spec.module_alias
        assert module is not None
        args = list(expr.args)
        if self.backend == "torch":
            dynamic_args = [arg for arg in args if not arg.is_number]
            if dynamic_args:
                first = dynamic_args.pop(0)
                result = self._print(first)
                args.remove(first)
                for arg in args:
                    value = self._print(arg)
                    if arg.is_number:
                        value = (
                            f"torch.as_tensor({value}, dtype={result}.dtype, "
                            f"device={result}.device)"
                        )
                    result = f"torch.{operation}({result}, {value})"
                return result
        result = self._print(args[0])
        for arg in args[1:]:
            result = f"{module}.{operation}({result}, {self._print(arg)})"
        return result


def _expression_tree(expr: sp.Expr | sp.MatrixBase) -> dict[str, object]:
    if isinstance(expr, sp.MatrixBase):
        return {
            "type": type(expr).__name__,
            "shape": list(expr.shape),
            "args": [
                [_expression_tree(value) for value in row] for row in expr.tolist()
            ],
        }
    node: dict[str, object] = {
        "type": getattr(expr.func, "__name__", type(expr).__name__),
    }
    if expr.args:
        node["args"] = [_expression_tree(argument) for argument in expr.args]
    else:
        node["value"] = str(expr)
    return node
