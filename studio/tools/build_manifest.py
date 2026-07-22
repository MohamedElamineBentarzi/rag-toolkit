"""Write `blocks.json` for local `npm run dev`.

The manifest *logic* lives in the shipped package (`rag_blocks.studio.manifest`)
so the `rag-blocks studio` CLI and this dev script share one source of truth.
This script is the thin dev-only wrapper that materializes it to a file for
Vite's dev server (which reads `public/blocks.json`). The CLI never needs the
file — it generates the manifest in memory.

Run from the repo root:

    python studio/tools/build_manifest.py           # writes studio/app/public/blocks.json
    python studio/tools/build_manifest.py --stdout   # print instead
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo importable when run from a source checkout without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_blocks.studio.manifest import build_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Studio's blocks.json")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = ap.parse_args()

    manifest = build_manifest()
    text = json.dumps(manifest, indent=2, sort_keys=True)
    if args.stdout:
        print(text)
        return
    out = Path(__file__).resolve().parents[1] / "app" / "public" / "blocks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(manifest['components'])} components)")


if __name__ == "__main__":
    main()
