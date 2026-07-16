# Contributing to rag-blocks

Thanks for your interest! This project is a learning vehicle for clean design as
much as a library, so **design discipline is a hard requirement, not a
nice-to-have.** The bar is high on purpose; this guide tells you exactly what
"done" means. The full rationale lives in [`AGENTS.md`](AGENTS.md) and
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Setup

```bash
git clone https://github.com/MohamedElamineBentarzi/rag-blocks
cd rag-blocks
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## The three checks (run before every push)

```bash
ruff check rag_blocks tests
mypy rag_blocks
pytest
```

CI runs all three on Python 3.10–3.13 (plus a Windows lane) and gates merges.
`ruff check` + `mypy` clean is a promise the README makes — keep it true.

## Tests

- **The default suite is fast and hermetic** — zero vendor deps, zero network,
  zero keys. That is the suite that gates PRs; keep it that way.
- **Real-stack tests are opt-in**, marked `@pytest.mark.integration`, and live
  in `tests/integration/`. Run them with `pytest -m integration` (they self-skip
  without the needed dependency or credential).
- **Tests ship WITH the feature, in the same PR. A component without tests does
  not exist.**
- **Every stage has a contract check** in `tests/contract_checks.py`
  (`assert_<stage>_contract`). Your new implementation's tests must call the
  matching one — that is how you inherit every behavioral guarantee the rest of
  the pipeline relies on, and how a reviewer knows you didn't skip an invariant.
- Test *our* logic, not vendors': extract pure functions and test them directly;
  fake through the registry seam production uses (see `tests/helpers.py`); never
  mock what you can fake through a designed seam.

## Definition of Done — a new component

1. Class with `kind`, `name`, `version`, optional nested `Config` dataclass;
   registered with `@registry.register`; wired into the subsystem `__init__.py`.
2. Implements exactly the stage ABC primitive(s); depends only on
   `core.contracts` + its own stage's abstractions (Dependency Inversion —
   a parser depends on `OcrEngine`, never on a concrete vendor).
3. Vendor deps: lazy import + a `pyproject` extra + an actionable `ImportError`.
4. Credential fields named for auto-redaction (`*_key`, `*_token`, …); env-var
   fallback.
5. Pure function of (config, inputs); heavy resources cached on the instance.
6. Streaming discipline if it produces data; provenance fields populated.
7. Hermetic tests including the stage contract check; an integration test only
   if a real vendor is involved (marked, env-gated).
8. Docstrings state the pattern and the *why*; README/ARCHITECTURE touched if
   user-visible.
9. `ruff check` clean, `mypy` clean, full suite green.

## The version-bump rule

A component's `version` is part of its fingerprint, which is a cache key.
**Any behavior change means bumping `version`** so stale caches invalidate
themselves. Renaming a field, changing a default, altering output — all count.

## Pull requests

- One logical change per PR; the PR template mirrors the Definition of Done.
- Keep the commit-message style already in the log (`Feat:`, `Fix:`, `Docs:`,
  `Refactor:`, `Chore:`, `CI:` prefixes). PRs are squash-merged, so the PR
  title becomes the commit.
- Be kind and precise in review — see the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs / proposing components

Use the issue templates (bug report / feature proposal). For questions, prefer
Discussions. For security issues, follow [SECURITY.md](SECURITY.md) — do not
open a public issue.
