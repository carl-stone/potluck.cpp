# Potluck engineering workflow

## Phase 0 resource safety

Before heavy work:

- Source `scripts/potluck-safe.sh` and run every build through `potluck_build`.
  It serializes builds and selects bounded jobs; do not use ad hoc `-j`, `-jN`,
  or `nproc` job counts.
- Use the 0.8B fixture by default for model-backed checks.
- Run 27B checks only while supervised, with no concurrent build.
- Assert available disk and memory with the safety guards before heavy work.
- Never run multi-device inference unsupervised.

## Commit discipline

Every Potluck-authored change lands as one or more atomic commits.

- One logical change per commit. A commit is a complete unit: it builds and
  passes `scripts/pre-push-check.sh` on its own.
- Never mix unrelated changes. Runtime behavior, test-contract fixes,
  documentation, and engineering tooling go in separate commits.
- Subject style: `potluck : <imperative summary>` for product changes,
  `dev : <imperative summary>` for process and documentation. ASCII only.
- Agents may prepare commits only on Carl's explicit instruction. Agent-made
  commits end with an `Assisted-by: <agent name>` trailer. Never
  `Co-authored-by:`.
- Never run `git push`, open a pull request, or comment upstream without
  Carl's explicit per-action approval.
