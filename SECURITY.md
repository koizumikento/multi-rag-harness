# Security Policy

## Supported Versions

`multi-rag-harness` is pre-1.0. Security fixes target the current `main` branch
until a stable release policy is published.

## Reporting a Vulnerability

Do not report security vulnerabilities in public issues.

Use GitHub private vulnerability reporting or a private maintainer contact path
when it is available for the published repository. Include:

- affected version or commit
- reproduction steps
- impact and affected data
- any relevant logs with secrets redacted

This project is local-first, but reports involving credential exposure, unsafe
MCP tool behavior, unintended file access, data corruption, or provenance
breakage are in scope.

## Handling Secrets

Keep API keys and credentials in environment variables. Do not put secrets in
TOML config files, checked-in examples, MCP plugin metadata, traces, durable
memory records, logs, tests, or docs.
