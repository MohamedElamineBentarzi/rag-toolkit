<!-- Thanks for contributing to rag-blocks! Keep PRs one logical change each. -->

## What & why

<!-- What does this change and why? Link any issue: Closes #123 -->

## Definition of Done

<!-- Tick what applies; a component without tests does not exist. -->

- [ ] Tests ship with the change (hermetic by default; integration only if a real vendor is involved, marked + env-gated)
- [ ] The relevant `assert_<stage>_contract` is called from the new tests
- [ ] `version` bumped on any behavior change (cache invalidation)
- [ ] `ruff check rag_blocks tests` clean
- [ ] `mypy rag_blocks` clean
- [ ] Full suite green (`pytest`)
- [ ] Docstrings state the pattern and the *why*; README/ARCHITECTURE touched if user-visible
- [ ] Secrets: credential fields named for auto-redaction; env-var fallback

## Notes for reviewers

<!-- Anything non-obvious: trade-offs, follow-ups, things you're unsure about. -->
