from dataclasses import dataclass
from typing import Literal, Mapping

Backend = Literal["python", "numpy", "torch", "jax"]


@dataclass(frozen=True)
class BackendSpec:
    argument_type: str
    scalar_return_type: str
    matrix_return_type: str
    module_alias: str | None
    imports: tuple[str, ...]
    function_aliases: Mapping[str, str]
    special_functions: Mapping[str, str]


_FUNCTION_ALIASES = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "cot": "cot",
    "sec": "sec",
    "csc": "csc",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "coth": "coth",
    "sech": "sech",
    "csch": "csch",
    "acos": "arccos",
    "acosh": "arccosh",
    "asin": "arcsin",
    "asinh": "arcsinh",
    "atan": "arctan",
    "atanh": "arctanh",
    "exp": "exp",
    "log": "log",
    "ceiling": "ceil",
    "floor": "floor",
    "sqrt": "sqrt",
    "ln": "log",
    "Sqrt": "sqrt",
}


BACKENDS: dict[Backend, BackendSpec] = {
    "python": BackendSpec("float", "float", "np.ndarray", None, (), {}, {}),
    "numpy": BackendSpec(
        "float | np.ndarray",
        "float | np.ndarray",
        "np.ndarray",
        "np",
        ("import numpy as np", "import math"),
        _FUNCTION_ALIASES,
        {
            "erf": "np.vectorize(math.erf)",
            "erfc": "np.vectorize(math.erfc)",
            "gamma": "np.vectorize(math.gamma)",
        },
    ),
    "torch": BackendSpec(
        "torch.Tensor",
        "torch.Tensor",
        "torch.Tensor",
        "torch",
        ("import torch",),
        _FUNCTION_ALIASES,
        {
            "erf": "torch.erf",
            "erfc": "torch.special.erfc",
            "gamma": "torch.special.gamma",
        },
    ),
    "jax": BackendSpec(
        "jax.Array",
        "jax.Array",
        "jax.Array",
        "jnp",
        ("import jax", "import jax.numpy as jnp", "import jax.scipy.special"),
        _FUNCTION_ALIASES,
        {
            "erf": "jax.scipy.special.erf",
            "erfc": "jax.scipy.special.erfc",
            "gamma": "jax.scipy.special.gamma",
        },
    ),
}
