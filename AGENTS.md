# Repository Guidelines

## Scope & Current Stage

一株 is an independent WeChat mini-program for plant identification and private observations, planned with Taro, React, TypeScript, and SCSS. Platform preparation is ongoing; do not generate business code without user approval. Never modify `/home/zhang/workspace/xdog` or join its workspace.

## Project Structure & Module Organization

- `docs/platform-setup.md`: completed steps, pending work, blockers, and next actions.
- `design/generated/`: existing visual experiments and generation metadata; preserve them.
- Planned after scaffolding: `src/pages/`, `components/`, `services/`, `hooks/`, `types/`, `styles/`, and `assets/`.
- Planned: `tests/` for automated checks; `cloudfunctions/` only if the backend decision requires it.
- `dist/` will contain generated WeChat output; never edit it manually.

Keep brand strings, design tokens, and asset provenance centralized.

## Build, Test, and Development Commands

No `package.json`, build scripts, or test runner exists yet. Do not present unconfigured commands as runnable.

When implementation is approved, verify official Taro documentation, scaffold a compatible stable stack, use the project-local CLI, and commit its lockfile. Document actual scripts for development, WeChat builds, type checking, and tests in `README.md`, including the developer-tool import directory.

## Coding Style & Naming Conventions

For new code, use two-space indentation, typed service boundaries, PascalCase React components, and camelCase functions. Prefer small shared components and SCSS tokens. No formatter or linter is configured; document selected tooling during scaffolding.

Use Taro components and mini-program APIs, not browser DOM globals. Keep layouts white, restrained, accessible, and photo-led; decoration must not obscure controls.

## Architecture & Testing Guidelines

Keep Species, Recognition, Plant, and Observation separate. Plant IDs remain stable across edits; associating an observation with an existing plant requires user confirmation.

Explicitly label mock results and document persistence. Preserve original recognition responses when users correct results.

No test framework or coverage threshold is configured. Adopt `*.test.ts` or `*.test.tsx`; prioritize ownership checks, stable IDs, association, idempotent retries, and reference-safe image deletion. Report type checks, builds, developer-tool checks, and real-device tests separately.

## Commit & Pull Request Guidelines

Git history is unavailable; conventions cannot be verified. Use imperative commits, optionally prefixed `docs:`, `feat:`, or `fix:`.

PRs should explain scope, link relevant issues, list executed checks and limitations, and include sanitized screenshots for UI changes.

## Security & Approval

Never commit secrets or private image links. Derive cloud identity from trusted login context and verify ownership. Request approval before paid resources, renewals, filing submissions, review submissions, or publication. Record platform progress without sensitive values.
