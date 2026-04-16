# Contributing

Who this is for: anyone — human or AI agent — working on a PR in `foveacast-training`. The goal of this file is to make sure every PR arrives with the code *and* the documentation it needs, first time through, instead of via a round of reviewer corrections.

If you are an AI assistant, [`CLAUDE.md`](CLAUDE.md) has the behavioural conventions specific to you; this file has the workflow conventions that apply regardless of who is typing.

---

## Before you start

- Read [`README.md`](README.md) (what the repo does and how to run it), [`ARCHITECTURE.md`](ARCHITECTURE.md) (end-to-end pipeline + the `.onnx` contract with Foveacast), and [`LEARNINGS.md`](LEARNINGS.md) (decisions, dead ends, and "oh, that's how that actually works" moments).
- For phased work, check [issue #1](https://github.com/khawkins98/foveacast-training/issues/1). It tracks the ten-phase plan from dataset fetch through Foveacast integration; each phase has an explicit gate. Know which gate you're opening before you write code.
- For decision gates (Phase 1 style), look for an open issue with the decision in its title. Making the decision is part of the work; don't skip to implementation.

## Branch + PR conventions

- **One phase per PR.** Issue #1 says "each phase is its own small PR" — don't bundle. If a single PR starts touching two phases, split it.
- **Branch naming:** `feat/phase-N-short-description` for phase work, `fix/...` / `chore/...` / `docs/...` for standalone commits outside the phase arc.
- **Stacked PRs:** if your branch depends on a still-open PR, target that PR as your base branch. When the base merges, GitHub retargets automatically to main. Prior examples: #4 stacked on #3, #6 stacked on #5.
- **Cross-link from issue #1** whenever a phase lands or closes. Tick the phase checkbox by editing the issue body (`gh issue edit 1 --body-file …`). Issue #1 is the single source of truth for phase progress.

## Commit hygiene

- **Conventional Commits:** `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:` with scopes where useful (`feat(msinet):`, `chore(phase-0):`, etc.).
- **One conceptual change per commit.** A commit that does three things at once is three commits. This matters because atomic commits make review faster, reverts precise, and `git bisect` actually useful.
- **Bodies explain *why*, not *what*.** The diff already shows what. Bodies should say why this change exists and — when the change was non-obvious — what alternative was rejected and why.
- **Co-author trailer** when Claude Code has contributed:
  ```
  Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
  ```
- **Pass commit messages via HEREDOC** so line breaks and quoted prose survive the shell.

## The documentation lockstep

A PR that lands real code almost always needs to touch docs. This checklist is the single load-bearing thing in this file; it encodes what past reviewer passes have caught. Run through it before marking a PR ready.

### Always

- [ ] Code changes come with tests where it is feasible to test them. Where it isn't, the PR body explains why.
- [ ] `.venv/bin/ruff check` passes across `src/`, `scripts/`, `tests/`.
- [ ] The full test suite passes (`.venv/bin/pytest`). A new commit must not break existing gates.
- [ ] A green summary with skips is NOT the same as a passing gate — if your work closes a phase gate, confirm the relevant test actually *ran* (check `-v` output), not just that nothing failed.

### When you change what a contributor runs

- [ ] **`README.md`** — new command, flag, extras group, or test file that contributors will invoke. Update the Setup section, the Tests section, and the Layout tree as applicable.

### When you change the pipeline shape or its contract

- [ ] **`ARCHITECTURE.md`** — flip status markers for modules you landed (`[landed]` in the Mermaid diagram, the module responsibility table, AND the phase-map row). Update the `.onnx` contract section if you changed input/output tensor shapes, dtypes, ranges, or channel order. Drift between prose and reality here poisons Phase 10 integration planning.

### When you close a phase

- [ ] **`LEARNINGS.md`** — dated entry. Not a changelog; a why-log. Covers any decision that took more than an hour to figure out, any dead end worth documenting, and any "oh, that's how that actually works" moment. Without this, the next contributor re-derives from scratch.
- [ ] Tick the phase checkbox on issue #1.
- [ ] Close any decision-gate issues you opened for that phase, referencing the LEARNINGS entry in the closing comment.

### When you change dataset handling

- [ ] **`data/README.md`** — anything about layout, splits, preprocessing expectations, upstream quirks, or resolved "choices deferred to Phase N".

### When you hit a new landmine

- [ ] **`CLAUDE.md` §Common pitfalls** — a bullet with enough context that the next session doesn't rediscover it. Load-bearing for long-lived research repos; tribal knowledge rots fast in solo-maintainer projects.

### When you cite new upstream work

- [ ] **`README.md` §Attribution chain.**
- [ ] **`CITATION.cff`** entry.
- [ ] Module docstring at the top of whichever file ports or wraps the upstream — named credit with paper reference and licence.

## Sub-agent reviews for non-trivial PRs

Spawning reviewer sub-agents — a technical writer + a DevRel — before merging any PR that lands a new module or changes the `.onnx` contract has caught real issues on Phase 2 (#5) and Phase 3 (#8). The pattern is worth using consistently:

1. Reviewers each get a self-contained prompt (target paths, what to evaluate, a word limit, a deliverable shape).
2. Run both in parallel — a single message with two `Agent` tool-use blocks.
3. Synthesize findings into three buckets: **must-fix** (land in the same PR), **nice-to-have** (follow-up PR), and **issue-worthy** (bigger work, separate issue).
4. Anything not fixed in the current PR goes into an issue with a specific scope — don't leave recommendations floating in PR comments.

Past prompts and results are in #5, #6, and #8 bodies if you need a template.

## Prose register

Run a humanizer pass on everything humans read — README, LEARNINGS, CHANGELOG (when it exists), PR bodies, commit messages, module docstrings, issue comments. Strip AI-register tells before committing: no "seamlessly", "delightfully", "robust solution", "leverage", "best-in-class", "simply", rule-of-three filler, em-dash overuse. The reference voice is the existing README and LEARNINGS — plain, specific, first-person when useful, why-focused.

If you are using Claude Code, the `/humanizer` skill does a pass for you. It is not magic; re-read afterwards.

## Risky actions

Confirm with the maintainer before:

- `git push --force` or anything that rewrites published history.
- `git reset --hard`, `git checkout .`, or bulk discard of uncommitted work.
- `rm -rf` anywhere inside the repo.
- Deleting or renaming `data/ueyes/`, `weights/`, or `runs/` — these take real time to rebuild.
- Modifying `pyproject.toml` dependency pins that force a full reinstall.
- Publishing to GitHub Releases, Zenodo, or HuggingFace.

Creating local commits, running `ruff` / `pytest`, and pushing a non-main branch are routine and don't need confirmation.

## Running the project locally

See [`README.md` §Setup](README.md#setup), [§Import the pretrained MSI-Net weights](README.md#import-the-pretrained-msi-net-weights), and [§Tests](README.md#tests). This file deliberately doesn't duplicate those commands — one source of truth, two links here.

## Proposing larger changes

If you're about to open a PR that spans phases, re-scopes the plan in issue #1, or changes the `.onnx` contract with Foveacast, raise an issue first and get alignment on the shape before writing code. Cost of asking is ~5 minutes; cost of re-doing a large change in the wrong shape is a day.

---

This file stays short on purpose. If a convention doesn't appear here or in [`CLAUDE.md`](CLAUDE.md), it's not a convention — propose it via an issue rather than enforcing it via review.
