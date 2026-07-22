"""The `rag-blocks studio` launcher: static serving + a live manifest route.

Hermetic — binds 127.0.0.1 on an ephemeral port, no external network. The bundled
build isn't present in a dev checkout, so a fake dist stands in for the static
files; the manifest route is the real thing.
"""
from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from rag_blocks import cli
from rag_blocks.studio import server


def test_create_server_errors_clearly_without_a_build(tmp_path):
    # The common editable-install case: no _dist/ yet. The error must say how.
    with pytest.raises(FileNotFoundError, match="npm run build"):
        server.create_server(dist=tmp_path)


def test_manifest_bytes_is_valid_json():
    data = json.loads(server.manifest_bytes())
    assert {"types", "stages", "components"} <= set(data)


@pytest.fixture()
def running_server(tmp_path):
    # A minimal fake static build so create_server is happy.
    (tmp_path / "index.html").write_text("<h1>Studio</h1>", encoding="utf-8")
    (tmp_path / "app.js").write_text("// bundle", encoding="utf-8")
    httpd = server.create_server(host="127.0.0.1", port=0, dist=tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def test_serves_the_static_index(running_server):
    status, body = _get(running_server + "/")
    assert status == 200
    assert b"Studio" in body


def test_blocks_json_is_the_live_generated_manifest(running_server):
    # Not read from disk — generated from this install's registry.
    status, body = _get(running_server + "/blocks.json")
    assert status == 200
    data = json.loads(body)
    assert any(c["name"] == "fixed" for c in data["components"])


def test_cli_has_a_studio_subcommand(capsys):
    # `rag-blocks` with no subcommand should exit non-zero and mention studio.
    with pytest.raises(SystemExit):
        cli.main([])
    assert "studio" in capsys.readouterr().err


def test_cli_studio_help_does_not_import_the_server(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["studio", "--help"])
    assert exc.value.code == 0
    assert "--port" in capsys.readouterr().out
