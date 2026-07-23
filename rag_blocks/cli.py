"""The `rag-blocks` command-line entry point.

Small on purpose: one subcommand today, `studio`, which launches the visual
pipeline builder. Kept as a subcommand dispatcher so more can be added without
changing the console-script wiring. Heavy imports are lazy — running `--help`
must not import the http server.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-blocks")
    sub = parser.add_subparsers(dest="command", required=True)

    studio = sub.add_parser("studio", help="launch the visual pipeline builder")
    studio.add_argument("--port", type=int, default=5173, help="port (default 5173)")
    studio.add_argument("--host", default="127.0.0.1", help="bind host")
    studio.add_argument(
        "--no-browser", action="store_true", help="don't open a browser"
    )

    args = parser.parse_args(argv)
    if args.command == "studio":
        from .studio.server import serve

        serve(host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0
    return 1  # unreachable: subcommand is required


if __name__ == "__main__":
    raise SystemExit(main())
