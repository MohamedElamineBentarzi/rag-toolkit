"""rag-blocks Studio: the optional visual pipeline builder.

Two stdlib-only pieces, no third-party dependency added to the core:
- `manifest.build_manifest()` — the registry introspected into UI data.
- `server.serve()` — serves the bundled static app (`rag-blocks studio`).

The React source and the checked-in dev manifest live outside the package, under
the repo's top-level `studio/`. Only what the shipped CLI needs is here.
"""

from .manifest import build_manifest

__all__ = ["build_manifest"]
