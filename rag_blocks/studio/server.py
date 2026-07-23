"""Serve the bundled Studio app locally — the guts of `rag-blocks studio`.

Studio ships as pre-built static files inside the wheel (`_dist/`, produced by
`npm run build` at release time). This module serves them with the stdlib
`http.server` — no Node, no third-party dep for the end user, just
`pip install rag-blocks`.

The one dynamic touch: `/blocks.json` is generated fresh at launch from *this*
install's registry (`build_manifest()`), not read from a checked-in file. So the
palette reflects exactly the components the user has — including their own
registered plugins — which a frozen manifest never could.
"""

from __future__ import annotations

import functools
import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .manifest import build_manifest


def dist_dir() -> Path:
    """The bundled static build, shipped as package data in the wheel."""
    return Path(__file__).resolve().parent / "_dist"


def manifest_bytes() -> bytes:
    return json.dumps(build_manifest(), indent=2, sort_keys=True).encode("utf-8")


class StudioHandler(SimpleHTTPRequestHandler):
    """Static file server for the app, with one live route: /blocks.json is the
    freshly generated manifest, so it never goes stale against the install."""

    manifest: bytes = b"{}"

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.split("?", 1)[0] == "/blocks.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.manifest)))
            self.end_headers()
            self.wfile.write(self.manifest)
            return
        super().do_GET()

    def log_message(self, *args: object) -> None:  # keep the console quiet
        pass


def create_server(
    host: str = "127.0.0.1", port: int = 5173, dist: Path | None = None
) -> ThreadingHTTPServer:
    """Build (but don't start) the server. `dist` is injectable for tests; in
    production it is the bundled `_dist/`.

    Raises a clear, actionable error when the static build is missing — the
    common case being an editable install where `npm run build` hasn't run.
    """
    dist = dist or dist_dir()
    if not (dist / "index.html").exists():
        raise FileNotFoundError(
            f"Studio's built assets aren't here ({dist}). This install has no "
            f"bundled build. From a source checkout, build them first:\n"
            f"  cd studio/app && npm install && npm run build\n"
            f"or use the dev server: cd studio/app && npm run dev"
        )
    handler = functools.partial(StudioHandler, directory=str(dist))
    StudioHandler.manifest = manifest_bytes()
    return ThreadingHTTPServer((host, port), handler)


def serve(
    host: str = "127.0.0.1", port: int = 5173, open_browser: bool = True
) -> None:
    """Start Studio and block until Ctrl-C."""
    httpd = create_server(host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"rag-blocks Studio -> {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
