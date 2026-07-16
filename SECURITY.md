# Security Policy

## Supported versions

`rag-blocks` is pre-1.0. Security fixes are applied to the latest released
minor version on PyPI. There is no long-term support branch yet.

| Version | Supported |
|---------|-----------|
| latest `0.x` | ✅ |
| older `0.x`  | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub's **[Private vulnerability reporting](https://github.com/MohamedElamineBentarzi/rag-blocks/security/advisories/new)**
(Security tab → "Report a vulnerability"). Include:

- a description of the issue and its impact,
- steps to reproduce (a minimal example is ideal),
- affected version(s).

You can expect an acknowledgement within a few days. Once a fix is ready we
will coordinate a release and credit you in the advisory unless you prefer to
remain anonymous.

## Scope notes

- The zero-dependency core runs on the standard library; most attack surface
  comes from optional vendor SDKs installed via extras.
- Credentials are read from config or environment variables and are redacted
  from `describe()`/`fingerprint()` output. If you find a path where a secret
  leaks into logs, trial records, or cache keys, that is in scope.
