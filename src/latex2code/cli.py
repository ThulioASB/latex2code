import argparse
import ast
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from latex2code import __version__
from latex2code.core import (
    Backend,
    LaTeXTranspilerError,
    inspect_latex,
    transpile_latex,
)


def _write_text_atomically(path: Path, content: str) -> None:
    """Write a file atomically so a failed write cannot truncate existing output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Transpile LaTeX mathematical expressions into executable Python functions."
    )
    parser.add_argument(
        "latex",
        nargs="?",
        type=str,
        help="The LaTeX expression wrapped in quotes, or '-' to read from stdin",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="formula",
        help="Name of the generated Python function (default: 'formula')",
    )
    parser.add_argument(
        "--no-types",
        action="store_true",
        help="Disable type hints in the generated function definition",
    )
    parser.add_argument(
        "--docstring",
        action="store_true",
        help="Embed the source LaTeX in the generated function's docstring",
    )
    parser.add_argument(
        "--backend",
        choices=("python", "numpy", "torch", "jax"),
        help="Generated code target (default: python)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write generated Python source to this file instead of stdout",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that generated output is syntactically valid Python without printing it",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print a JSON report with the parsed expression, variables, warnings, and code",
    )
    parser.add_argument(
        "--map-variable",
        action="append",
        metavar="SYMBOL=NAME",
        help="Rename a parsed variable to a Python argument name (repeatable)",
    )
    parser.add_argument(
        "--args",
        metavar="NAMES",
        help="Comma-separated argument order using parsed symbol names, e.g. x,w,b",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Read expressions interactively until 'quit' or end-of-file",
    )

    args = parser.parse_args()

    if args.latex is None and not args.interactive:
        parser.print_help()
        return
    if args.interactive and args.latex is not None:
        parser.error("--interactive cannot be combined with a positional expression")
    if args.inspect and args.output:
        parser.error("--inspect cannot be combined with --output")
    if args.check and (args.inspect or args.output):
        parser.error("--check cannot be combined with --inspect or --output")
    if args.check and args.interactive:
        parser.error("--check cannot be combined with --interactive")
    if args.interactive and args.output:
        parser.error("--interactive cannot be combined with --output")

    variable_map = {}
    for item in args.map_variable or ():
        symbol, separator, name = item.partition("=")
        if not separator or not symbol or not name:
            parser.error(f"invalid --map-variable value {item!r}; expected SYMBOL=NAME")
        if symbol in variable_map:
            parser.error(f"variable {symbol!r} was mapped more than once")
        variable_map[symbol] = name

    argument_order = None
    if args.args is not None:
        argument_order = [name.strip() for name in args.args.split(",") if name.strip()]
        if not argument_order:
            parser.error("--args needs at least one name, e.g. --args x,y")

    latex_input = args.latex
    if args.interactive:
        _run_interactive(
            function_name=args.name,
            type_hints=not args.no_types,
            docstring=args.docstring,
            backend=args.backend,
            inspect=args.inspect,
            variable_map=variable_map,
            argument_order=argument_order,
        )
        return

    if latex_input == "-":
        latex_input = sys.stdin.read().strip()

    try:
        if args.check:
            source = transpile_latex(
                latex_input,
                function_name=args.name,
                type_hints=not args.no_types,
                docstring=args.docstring,
                backend=args.backend,
                variable_map=variable_map,
                args=argument_order,
            )
            ast.parse(source, filename="<latex2code-generated>")
            print("Generated Python syntax is valid.")
        elif args.inspect:
            details = inspect_latex(
                latex_input,
                function_name=args.name,
                type_hints=not args.no_types,
                docstring=args.docstring,
                backend=args.backend,
                variable_map=variable_map,
                args=argument_order,
            )
            print(json.dumps(asdict(details), indent=2))
        else:
            source = transpile_latex(
                latex_input,
                function_name=args.name,
                type_hints=not args.no_types,
                docstring=args.docstring,
                backend=args.backend,
                variable_map=variable_map,
                args=argument_order,
            )
            if args.output:
                _write_text_atomically(args.output, source)
            else:
                print(source)
    except LaTeXTranspilerError as err:
        print(f"\n[Error] {err}", file=sys.stderr)
        sys.exit(1)
    except SyntaxError as err:
        print(f"\n[Error] Generated Python failed syntax validation: {err}", file=sys.stderr)
        sys.exit(1)
    except OSError as err:
        print(f"\n[Error] Could not write output: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[Unexpected Error] {err}", file=sys.stderr)
        sys.exit(1)


def _run_interactive(
    *,
    function_name: str,
    type_hints: bool,
    docstring: bool,
    backend: Backend | None,
    inspect: bool,
    variable_map: dict[str, str],
    argument_order: list[str] | None = None,
) -> None:
    mode = "inspection" if inspect else "code generation"
    print(
        f"latex2code interactive ({mode}); enter 'quit' to exit, "
        "or ':begin' / ':end' for a multiline expression."
    )
    multiline: list[str] | None = None
    while True:
        try:
            expression = input("latex2code> " if multiline is None else "...> ").strip()
        except EOFError:
            if multiline is not None:
                print("Discarded unfinished multiline expression.", file=sys.stderr)
            print()
            return
        if multiline is not None:
            if expression == ":end":
                _print_interactive_expression(
                    "\n".join(multiline),
                    function_name=function_name,
                    type_hints=type_hints,
                    docstring=docstring,
                    backend=backend,
                    inspect=inspect,
                    variable_map=variable_map,
                    argument_order=argument_order,
                )
                multiline = None
            elif expression == ":cancel":
                multiline = None
                print("Cancelled.")
            else:
                multiline.append(expression)
            continue
        if expression == ":begin":
            multiline = []
            continue
        if expression.lower() in {"quit", "exit"}:
            return
        if not expression:
            continue
        _print_interactive_expression(
            expression,
            function_name=function_name,
            type_hints=type_hints,
            docstring=docstring,
            backend=backend,
            inspect=inspect,
            variable_map=variable_map,
            argument_order=argument_order,
        )


def _print_interactive_expression(
    expression: str,
    *,
    function_name: str,
    type_hints: bool,
    docstring: bool,
    backend: Backend | None,
    inspect: bool,
    variable_map: dict[str, str],
    argument_order: list[str] | None = None,
) -> None:
    try:
        if inspect:
            details = inspect_latex(
                expression,
                function_name=function_name,
                type_hints=type_hints,
                docstring=docstring,
                backend=backend,
                variable_map=variable_map,
                args=argument_order,
            )
            print(json.dumps(asdict(details), indent=2))
        else:
            source = transpile_latex(
                expression,
                function_name=function_name,
                type_hints=type_hints,
                docstring=docstring,
                backend=backend,
                variable_map=variable_map,
                args=argument_order,
            )
            print(source, end="" if source.endswith("\n") else "\n")
    except LaTeXTranspilerError as err:
        print(f"[Error] {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
