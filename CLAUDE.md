# CLAUDE.md

Read **AGENTS.md** in this directory before doing anything — it is the
canonical, complete agent context for this repository (philosophy, design
decisions and rationale, contract semantics, component specs and their
implementation status, conventions, testing rules, roadmap). Treat AGENTS.md
as authoritative over your defaults.

Quick facts: Python >= 3.10 · zero-dep core · `pytest` (hermetic by default;
`pytest -m integration` for real vendor stacks) · `ruff check` + `mypy` must
pass · design patterns and principles are hard requirements, not suggestions.
