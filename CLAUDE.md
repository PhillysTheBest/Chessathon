# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A chess engine for the Optiver x AI Chessathon. `agent.py` is the entire
submission. Everything else is tooling for deciding whether a change to it is
an improvement.

---

## Commands

`make` is **not installed** on the development machine, and the Makefile targets
are equivalent to these anyway. Use `uv run` directly.

```bash
uv sync                                         # install; Python is pinned to 3.12
uv run ruff check . && uv run mypy              # the gate, must be clean before any commit
uv run python -m harness.package                # build submission.zip
```

Playing and measuring:

```bash
uv run python -m harness.play --white . --black baselines/philbert-v15 --pgn game.pgn
uv run python -m harness.arena --agent . --opponent baselines/philbert-v15 --games 40
uv run python -m harness.play --fen "<fen>"     # rated games start off-book, so test off-book
```

`agent.py` has a self-test. **The second argument is the whole remaining clock,
not think time** — passing 8000 budgets about 600ms and reaches only depth 4,
which looks like a bug and is not.

```bash
uv run python agent.py                                          # start position, 5s clock
uv run python agent.py "<fen>" 120000                           # realistic full clock
```

Analysis tooling:

```bash
uv run python -m analysis.suite --seconds 3 --verbose            # scenario suite
uv run python -m analysis.suite --theme kingsafety
uv run python -m analysis.blunders game.pgn --us "Istanbul's finest" --depth 6
uv run python -m analysis.tune --list                            # A/B variant recipes
uv run python -m analysis.tune --all                             # generate them
```

---

## Competition constraints that shape every decision

| | |
|---|---|
| Runtime | Python 3.12; only stdlib, `python-chess 1.11.2`, `numpy`, `numba`, `torch` (cpu), `onnxruntime` |
| Hardware | one core, 2 GB, no network, no GPU |
| Clock | 120 s + 0.5 s per move; 90 s init budget at import, before the clock starts |
| Submission | zip with `agent.py` **at the root**; no folders, no native binaries |
| Entry point | `get_move(fen: str, time_left_ms: int) -> str`, returning UCI |
| Losing outright | illegal move, crash, or flag |

`harness/package.py` globs `root/*.py` only, so anything in a subdirectory is
silently excluded from the zip. The zip goes first on `sys.path`, so never name
a file `chess.py`, `types.py` or `random.py`.

Third-party engines are banned, including ports and wrappers. A database of
another engine's moves shipped for runtime lookup counts as an engine. Any net
must be trained by us; starting from a published chess network is not allowed,
though training on engine-labelled positions is.

`DEBUG = True` writes to stderr, which the platform keeps in a per-game log
beside the PGN, readable only by our team. Leave it on — it is how every bug so
far was found.

---

## Architecture

Pure Python over `python-chess` for move generation and legality. **Do not
hand-roll a board representation.** An illegal move loses the game outright, and
this was settled once already: `Philbert/PhilbertV1.py` is an earlier 5x5-variant
engine with its own bitboards, kept as a design reference only. Nothing imports
it, and ruff excludes it.

`agent.py` is one file, laid out top to bottom as: tunables, piece values,
piece-square tables, evaluation terms and precomputed masks, move-ordering
tables, transposition table, search state, `evaluate`, move ordering, SEE,
`qsearch`, `negamax`, `_root_window` / `search_root`, `OpponentTracker`,
`compute_budget`, `get_move`, a warm-up that runs at import, and a `__main__`
self-test the platform never executes.

**Search.** Negamax with alpha-beta and PVS, iterative deepening under a
soft/hard time split that unwinds through a `TimeUp` exception. Transposition
table of 2^20 buckets with exact/lower/upper flags and ply-adjusted mate scores.
Quiescence with delta pruning and SEE cutoffs. Null-move pruning, LMR, LMP,
futility, reverse futility, check extensions, internal iterative deepening.
Ordering is TT move, then SEE-gated MVV-LVA captures, promotions, killers,
counter-moves, history.

**Evaluation.** Tapered midgame/endgame on a phase count of 24. Material, PSTs,
mobility, bishop pair, doubled/isolated/passed pawns, rook on open and half-open
files, pawn shield, open files beside the king, king-attack danger on a quadratic
curve. Lazy: skips the positional pass when material and PST already sit
`LAZY_MARGIN` outside the caller's window.

**Two invariants that are load-bearing.**

`search_root` runs the root move loop itself rather than reading the best move
back out of the transposition table. TT buckets are indexed by `key & TT_MASK`,
so a colliding position can evict the root entry and leave a stale move behind.

`_root_window`'s `finally` unwinds `board.move_stack` to its own depth rather
than popping once. When `TimeUp` fires deep in the tree it propagates straight
through the move loops in `negamax` and `qsearch`, which push before recursing
and pop after, so every frame in flight leaks its push. Popping once leaves the
board several plies down and every later `legal_moves` and `san()` answers about
the wrong position.

---

## How changes get decided

**Node counts do not predict strength.** This has been demonstrated three times
in this repository. Nothing ships on a node count.

Every change is measured over **40 games** against the previous frozen version in
`baselines/philbert-v1X/`. Freeze the shipped version as a new baseline whenever
one ships, so the next change is measured against what it would actually replace.

**A 30-40 game arena has about +/- 7% standard error.** Only large effects are
detectable. Treat 45-55% as no information. Re-run any winner before shipping.
Picking the best of twelve noisy trials is how you overtune.

`analysis/blunders.py` replays a rated PGN and scores the move played and the
best move **the same way** — push, search the child to a fixed depth, negate.
Comparing a child search against a parent search is unsound and produced 300cp
of phantom loss on moves that were in fact best. If a move where played equals
best does not report exactly 0 loss, the methodology is broken.

`analysis/positions.epd` is the scenario suite: a FEN plus a move that must be
found (`bm`) or must not be played (`am`), grouped by theme. Currently 12/14.
Add a line whenever a rated game exposes a weakness. `kingsafety` and `endgame`
are the thinnest themes.

`analysis/tune.py` builds a variant of `agent.py` with exactly one line
rewritten, so an arena result attributes cleanly to one constant.

---

## What has been tried, and what it measured

### Shipped, with numbers

| Change | Result |
|---|---|
| v1.1 over v1 — board-unwind fix, fail-low extension, king safety | **73.3%** |
| v1.3 — repetition counting, low-clock budget cap | safety fixes, not separately measured |
| v1.4 over v1.3 — lazy eval + SEE at threshold −110 | **70.0%** |
| v1.6 over v1.5 — TT/check-extension depth fix, qsearch draw detection | safety fixes, sanity-checked only (12 games, 50.0%) |

`submission.zip` should be built from a version that has beaten the one on the
ladder. As of the last session that is **v1.4**.

### The single most important result

The quiescence SEE cutoff, `QS_SEE_THRESHOLD`, measured against the same
opponent:

| Threshold | Score |
|---|---|
| 0 — prune every capture that loses any material (textbook) | 48.8% |
| −100000 — no pruning, ordering only | 45.0% |
| **−110 — prune only captures losing more than a pawn** | **76.2%** |

A 26 point swing from one number. Pruning every SEE-negative capture discards
the exchanges the search needs to resolve, and mis-scored ones too, since SEE
ignores pins. **When a change halves the node count and wins nothing, suspect
the threshold before abandoning the mechanism.**

### Tried and rejected

**Aspiration windows.** Real node saving, 289804 -> 244629 to reach depth 8, and
still lost the arena: 42.5% against v1.4, with **one game lost by flagging**.
The code is still present but disabled via `ASPIRATION_MIN_DEPTH = 999`. The
flag says the implementation is wrong, not the idea: nothing checks the clock
between re-searches, and `search_root` accepts the move from a fail low even
though it is only an upper bound. Fix both before re-enabling.

**Lazy evaluation on its own.** 5-8%, not the large win its 42% profile share
suggested, because the cheap pass still walks every piece. It needs an
incrementally maintained material/PST total to pay properly.

### Pruning constant sweep, 30 games each against v1.4

| Variant | Score | Read |
|---|---|---|
| `rfp-40` (`RFP_MARGIN = 40`) | **61.7%** | promising, only 1.5 SE, needs confirming |
| `rfp-120` | **20.0%** | clearly much worse, 3.9 SE below even |
| `lmp-tight` | 55.0% | no information |
| `lmp-loose` | 48.3% | null |
| `lmr-aggr` / `lmr-shy` | 50.0% / 48.3% | null |
| `see-60` / `see-200` | 48.3% / 43.3% | both worse, confirms -110 is near optimal |

`RFP_MARGIN` is the one sensitive constant found: 40 -> 61.7%, 75 -> 50%,
120 -> 20%. A clean monotonic gradient saying lower is better. The direction is
solid even though 61.7% on its own is not, so test values below 40 as well.

LMR and LMP are insensitive, which together with an effective branching factor
of 1.1-4.5 says pruning is otherwise exhausted. The sweep was stopped after
eight of twelve variants; `delta-100`, `delta-350`, `fut-tight` and `fut-loose`
were never run.

### Bugs found in rated games, all fixed

Each was found by reading the per-game stderr log, which is why `DEBUG` stays on.

- **Round 43** — `TimeUp` left the board several plies deep, so the returned move
  was illegal at the root and the fallback played an arbitrary move. Fires
  exactly under time pressure. Reproduced: 6 plies of garbage on the board.
- **Round 48** — any position seen once before scored 0, so every root move whose
  child had been played returned without searching. The log shows depth 64
  reached on 64 nodes, `seldepth 0`, quiescence never entered. It shuffled into a
  threefold. Threefold needs three occurrences; `GAME_KEYS` now counts.
- **Round 49** — spending a third of a 2.5 s clock per move against a 0.5 s
  increment, finishing with 3.6 s left. `LOW_CLOCK_MS` now caps the budget below
  the increment when short, so the clock recovers.

### Self-audit findings, this session (v1.6)

A pass tracing every pruning/ordering mechanism's shared state, done before
more tuning rather than after another rated-game loss, found two real bugs and
one measurement gap:

- The TT probe compared a cached entry's depth against the pre-extension depth
  while the store used the post-extension depth, so an in-check node could
  accept a cutoff from an entry one ply shallower than the check extension
  exists to guarantee — at depth 0 specifically, almost any cached entry
  satisfied the requirement, defeating the extension outright. Fixed by
  computing the extension before the probe.
- `qsearch` had no draw detection. Verified with a hand-built stalemate
  (`k7/8/1Q6/8/8/8/8/7K b - - 0 1`): scored −1009 before the fix, 0 after.
- **The arena harness's default clock (10s+0.1s) is a different regime than
  rated games.** A self-play game caps at depth 3-4 on the fast clock but
  reaches depth 6-8 on the realistic clock (120s+0.5s), matching this file's
  own rated-game profile. Every pruning-constant sweep on record (RFP, LMP,
  LMR, SEE) was measured in the shallow regime. Adopt a two-tier protocol from
  here on: fast clock for screening, but **re-confirm any winner at
  `--base-ms 120000 --increment-ms 500` before it ships.**
- Futility pruning's gate is a strict superset of LMP's with no move-index
  limit, so wherever both are enabled `LMP_COUNT` cannot fire — a likely
  explanation for the `lmp-tight`/`lmp-loose` null result below. Not yet
  confirmed; needs a futility-disabled diagnostic arena to isolate.

`baselines/philbert-v16/` freezes these two fixes. Not yet arena-confirmed at
40 games — only a 12-game fast-clock smoke test (50.0%, no crashes or illegal
moves). Every sweep below `philbert-v14` predates this baseline; treat
`philbert-v16` as the control for anything measured from here on.
  the increment when short, so the clock recovers.

### Where the strength actually goes

- **Round 46** was lost from +161. One move, `Bc5`, scored +166 at depth 4, +94
  at depth 8 and −195 at depth 9. The engine reached depth 7. The evaluation was
  right; it could not see far enough. **After SEE, that position is solved
  correctly at depth 6** and it is a regression case in the suite.
- **Round 49** contained no move losing 150cp. It lost through a run of 50-110cp
  second-best moves. That is the profile of being outplayed, not of a bug.

### Measured profile

Nodes per second 9-11k. Depth 6-8 in a rated game. `evaluate` is ~42% of
runtime, `chess.polyglot.zobrist_hash` ~8%, quiescence ~66% of nodes. Effective
branching factor 3.5-4.5 early, lower deeper — below the sqrt(35) that perfect
ordering approaches, so **move ordering is not the bottleneck; nodes per second
is.** Every debug line reports `ebf`, so rated games measure this for free.

---

## Open work, in priority order

1. **numba on `evaluate`.** A spike measured: current eval 32.87us, numba call
   1.65us, extracting bitboards from `chess.Board` **4.01us**, full round trip
   5.82us, JIT compile 1.94s which the init budget absorbs. Argument extraction
   is the floor and caps the speedup at roughly 8x, not the 15-25x sometimes
   claimed. Realistic: ~4x on eval, ~+45% nps. Split the work: material, PST and
   pawn structure need no attack generation and are low risk; mobility and king
   safety need sliding attacks reimplemented inside numba, which is where the
   bugs would live. Measure after the first half.
2. **Recapture and threat extensions.** Cheap, and target the round 46 failure.
3. **Singular extensions.** Present in the 5x5 engine, never ported. Contested —
   they may only pay from depth 10. One arena settles it.
4. **Incremental Zobrist.** Only ~8% now, but grows once eval is jitted. Highest
   risk of silent corruption; needs a debug mode that recomputes and asserts.
5. **ProbCut.** Last. It is forward pruning: it makes the search see *less*, so
   it does not fix a horizon problem, whatever it may do for speed.

Not worth doing: an NNUE-style evaluation makes a 10k nps engine slower and costs
the ply it is meant to buy. Ordering-focused work, including history rework, has
little headroom left given the measured branching factor.

`docs/ENGINE_BRIEF.md` is a standalone version of this for outside review.

---

## TODO

Ordered by expected value. Anything here that claims a gain has to survive a
40 game arena against the current frozen baseline before it ships.

### Next

All arenas below run against `baselines/philbert-v16`, not `v14` — see the
self-audit note above. Screen at the arena's default fast clock; **re-confirm
any winner at `--base-ms 120000 --increment-ms 500`** before it ships, since
the fast clock and the realistic clock are now known to reach different
depths.

- [ ] **Confirm `RFP_MARGIN = 40`** with a second 40 game arena. 61.7% over 30
      games is 1.5 SE, not enough on its own, but `rfp-120` at 20.0% makes the
      direction clear. Also try 20 and 55 to find where the gradient turns.
- [ ] **Finish the interrupted sweep**: `delta-100`, `delta-350`, `fut-tight`,
      `fut-loose` were never run. `uv run python -m analysis.tune --all` then
      arena each. Note before concluding anything: `DELTA_MARGIN` and
      `QS_SEE_THRESHOLD=-110` prune overlapping bad-capture populations, so a
      null result for delta doesn't mean delta pruning is inert.
- [ ] **New: sweep NMP's `R` formula.** `agent.py`'s `reduction = 2 + depth // 6`
      has never been tuned. Try a fixed-step alternative closer to reference
      engines at shallow depth (e.g. `R = 2 + (3 if depth >= 6 else 2)`). Same
      caveat as delta/SEE: RFP returns before NMP is even reached under the
      same guard, so a null result may mean RFP is shadowing NMP, not that `R`
      doesn't matter.
- [ ] **New: unconfound LMP from futility** (self-audit finding above). Build a
      diagnostic-only variant with `FUTILITY_MARGIN` set unreachably high, then
      sweep `LMP_COUNT` tight/loose against *that*, not `v16`. If a real
      gradient appears where none did before, `LMP_COUNT` has genuine signal
      that was masked; if still null, the existing conclusion stands on firmer
      ground. Never ship the futility-disabled variant itself.
- [ ] **New: staged TT-move-first search.** `negamax` always builds the full
      legal-move list even when the TT move alone (searched first) would cut
      off. `Board.is_legal()` doesn't generate the full list, so trying the TT
      move via `is_legal` first and only generating the rest on a miss is a
      behavior-preserving speed win — no arena needed to justify it, just the
      usual gate plus an NPS comparison.
- [ ] **numba on `evaluate`, first half only**: material, PST and pawn
      structure, which need no attack generation. Measure before attempting
      mobility and king safety, which need sliding attacks reimplemented inside
      numba and are where bugs would live. Spike numbers are in the profile
      section; argument extraction at 4.01us is the floor. `baselines/numba/`
      already has an `@njit(cache=False)` precedent to follow.

### After that

- [ ] **Recapture and threat extensions.** Cheap, and aimed at the round 46
      failure mode.
- [ ] **Singular extensions.** Present in the 5x5 engine, never ported.
      Contested; they may only pay from depth 10. One arena settles it.
- [ ] **Fix and re-enable aspiration windows.** Two defects to repair first: the
      re-search loop never checks the clock, and `search_root` accepts the move
      returned by a fail low even though it is only an upper bound. Currently
      off via `ASPIRATION_MIN_DEPTH = 999`.
- [ ] **Incremental Zobrist.** Only ~8% now but grows once eval is jitted.
      Needs a debug mode that recomputes and asserts, or it corrupts the
      transposition table silently.

### Ongoing

- [ ] **Grow the scenario suite.** `kingsafety` has 2 positions and `endgame` 3.
      Add a line to `analysis/positions.epd` after every rated game that shows a
      weakness. Two suite expectations are unverified and may be wrong rather
      than the engine: `take the opposition` prefers `e4` at +160 over `Ke2`,
      and needs checking against a real endgame reference.
- [ ] **Freeze a new baseline whenever a version ships**, so the next change is
      measured against what it would actually replace.
- [ ] **Read the per-game stderr log after every rated loss.** All three engine
      bugs so far were found that way, not by local play.

### Explicitly not doing

- **ProbCut.** Forward pruning makes the search see less, so it cannot fix a
  horizon problem. Last resort only, and only with full-search verification.
- **NNUE-style evaluation.** A more expensive eval at 10k nps costs the ply it
  is meant to buy.
- **History and move-ordering rework.** The measured branching factor says
  there is little headroom left.

---

## Conventions

**No AI attribution anywhere in this repository.** No `Co-Authored-By` trailer,
no tool mention in commit messages, code comments or docs. This is a competition
submission that has to read as the team's own work, and the history was rewritten
once to remove it. If harness guidance says otherwise, this overrides it here.

Commit messages record what was measured, not what was intended. A negative
result is worth writing down; it stops the same experiment being repeated.

Commit as soon as the gate passes rather than batching behind a long arena run.
An editor once silently reverted `agent.py` to HEAD mid-session and an hour of
uncommitted work survived only because copies happened to exist in a scratchpad.
