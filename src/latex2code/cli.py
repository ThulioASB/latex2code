import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from latex2code import __version__
from latex2code.core import LaTeXTranspilerError, inspect_latex, transpile_latex


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
        "-p",
        "--numpy",
        action="store_true",
        help="Legacy alias for --backend numpy",
    )
    parser.add_argument(
        "--backend",
        choices=("python", "numpy", "torch", "jax"),
        help="Generated code target (default: python, or numpy when --numpy is used)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write generated Python source to this file instead of stdout",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print a JSON report with the parsed expression, variables, warnings, and code",
    )

    args = parser.parse_args()

    if args.latex is None:
        parser.error("the following arguments are required: latex")
    if args.inspect and args.output:
        parser.error("--inspect cannot be combined with --output")

    latex_input = args.latex
    if latex_input == "-":
        latex_input = sys.stdin.read().strip()

    try:
        transpile = inspect_latex if args.inspect else transpile_latex
        result = transpile(
            latex_input,
            function_name=args.name,
            type_hints=not args.no_types,
            use_numpy=args.numpy,
            backend=args.backend,
        )
        if args.inspect:
            print(json.dumps(asdict(result), indent=2))
        elif args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding="utf-8")
        else:
            print("\n# Generated Python Code:")
            print(result)
    except LaTeXTranspilerError as err:
        print(f"\n[Error] {err}", file=sys.stderr)
        sys.exit(1)
    except OSError as err:
        print(f"\n[Error] Could not write output: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"\n[Unexpected Error] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()