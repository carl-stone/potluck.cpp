# Potluck engineering workflow

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
