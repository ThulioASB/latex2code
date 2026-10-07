import argparse
import sys
from pathlib import Path
from latex2code.core import LaTeXTranspilerError, transpile_latex


def main():
    parser = argparse.ArgumentParser(
        description="Transpile LaTeX mathematical expressions into executable Python functions."
    )
    parser.add_argument(
        "latex",
        type=str,
        help="The LaTeX expression wrapped in quotes, or '-' to read from stdin",
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
        help="Use NumPy arrays and functions",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write generated Python source to this file instead of stdout",
    )

    args = parser.parse_args()

    latex_input = args.latex
    if latex_input == "-":
        latex_input = sys.stdin.read().strip()

    try:
        code = transpile_latex(
            latex_input,
            function_name=args.name,
            type_hints=not args.no_types,
            use_numpy=args.numpy,
        )
        if args.output:
            args.output.write_text(code, encoding="utf-8")
        else:
            print("\n# Generated Python Code:")
            print(code)
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