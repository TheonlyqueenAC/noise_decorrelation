# python
import argparse
from typing import Any


def greet(name: str) -> str:
    """Return a greeting for `name`."""
    return f"Hi, {name}"


def main(argv: Any = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = argparse.ArgumentParser(description="Simple greeting script.")
    parser.add_argument("-n", "--name", default="PyCharm", help="Name to greet")
    args = parser.parse_args(argv)
    print(greet(args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
