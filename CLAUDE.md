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

Playing and measuring (`baselines/philbert-v1X/` is illustrative syntax below —
use the highest version number under `baselines/` for the actual current
baseline):

```bash
uv run python -m harness.play --white . --black baselines/philbert-v15 --pgn game.pgn
uv run python -m harness.arena --agent . --opponent baselines/philbert-v15 --games 40
uv run python -m harness.arena --agent . --opponent baselines/philbert-v15 --games 40 --sprt
uv run python -m harness.play --fen "<fen>"     # rated games start off-book, so test off-book
```

`--sprt` stops an arena early once the result is decisively one-sided (a Wald
SPRT approximation), instead of always playing every game; `--games` is still
the hard cap. Bounds default to `elo0=0, elo1=40`, sized for OUR sample sizes
(the arena's own noise floor at n=40 is roughly +/-55 elo) rather than the
tiny elo0=0/elo1=5 bounds big engine-testing projects use with thousands of
games. Use it to screen — a clearly bad change (~20% score, like `rfp-120`'s
historical 20.0%) resolves in ~25 games instead of running the full count,
and a clearly great one (~75%, like the SEE threshold's 76.2%) resolves
around 70-80. **A genuine toss-up correctly refuses to resolve early and runs
to the `--games` cap** — that's by design, not a bug: forcing a call on a
real coin flip would be worse than the noise it's meant to filter out. Off by
default; every existing recorded command above keeps its exact fixed-N
behavior.

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
it, and ruff excludes it. Where speed matters, code reads `python-chess`'s raw
bitboard attributes directly (`board.pawns`, `board.occupied_co[color]`, ...)
rather than the slower object-returning APIs (`piece_at`, `pieces`) — but the
board itself, move generation, and legality all stay `python-chess`'s.

`agent.py` is one file, laid out top to bottom as: tunables, piece values,
piece-square tables, evaluation terms and precomputed masks, numba-jitted
material/PST/mobility helpers, move-ordering tables, transposition table,
search state, `evaluate`, move ordering, SEE, incremental-Zobrist helpers,
`qsearch`, `negamax`, `_root_window` / `search_root`, `OpponentTracker`,
`compute_budget`, `get_move`, a warm-up that runs at import, and a `__main__`
self-test the platform never executes.

**Search.** Negamax with alpha-beta and PVS, iterative deepening under a
soft/hard time split that unwinds through a `TimeUp` exception. Transposition
table of 2^20 buckets with exact/lower/upper flags and ply-adjusted mate scores.
Zobrist keys are computed incrementally in the hot recursive call sites
(threaded as an optional argument that falls back to a full
`chess.polyglot.zobrist_hash` recompute wherever not supplied — no call site
can produce a *wrong* key, only a slower-but-correct one). A cached TT move is
tried via the cheap `board.is_legal()` check before the full legal-move list is
generated at all; only a miss pays for full generation and ordering. Quiescence
with delta pruning and SEE cutoffs, plus its own draw/stalemate detection.
Null-move pruning, LMR, LMP, futility, reverse futility, check extensions, a
forced-reply extension (one legal move costs nothing extra to search one ply
deeper), internal iterative deepening. Ordering is TT move, then SEE-gated
MVV-LVA captures, promotions, killers, counter-moves, history.

**Evaluation.** Tapered midgame/endgame on a phase count of 24. Material, PSTs
and the bishop pair are numba-jitted (bit-exact against the prior pure-Python
version). Mobility and the king-zone-attack term are also numba-jitted, via
hand-rolled ray-casting for sliding pieces — `python-chess`'s own sliding-attack
tables are Python dicts numba can't index efficiently, so this isn't reused
directly, only verified against it (`board.attacks_mask()`) per-square before
being trusted. Doubled/isolated/passed pawns, rook on open and half-open files,
pawn shield, open files beside the king, and king-attack danger on a quadratic
curve stay pure Python — cheap bitwise-mask operations with no attack
generation, so jitting them doesn't pay off the same way. Lazy: skips the
positional pass when material and PST already sit `LAZY_MARGIN` outside the
caller's window.

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
Picking the best of twelve noisy trials is how you overtune. `--sprt` (see
Commands) can shorten a clearly one-sided result but does not change this bar —
a genuine toss-up still runs the full count.

**A proof of bit-exact correctness can substitute for an arena, but only for
changes that provably cannot alter *which* nodes get searched or *what*
decision gets made at a given depth** — a numba rewrite of a pure function is
the clean case: same inputs, same outputs, only faster, verified by comparing
outputs across thousands of positions. The moment a change touches *when* a
cutoff fires, *which* moves get generated, or *what* gets cached, node
visitation order can shift, and LMR/NMP/futility are approximate enough that a
different cache population can change a score by tens of centipawns even with
identical logic — that class of change still needs the arena. (The staged
TT-move-first search looked like the safe case and wasn't; see "What has been
tried" below.)

`analysis/blunders.py` replays a rated PGN and scores the move played and the
best move **the same way** — push, search the child to a fixed depth, negate.
Comparing a child search against a parent search is unsound and produced 300cp
of phantom loss on moves that were in fact best. If a move where played equals
best does not report exactly 0 loss, the methodology is broken.

`analysis/positions.epd` is the scenario suite: a FEN plus a move that must be
found (`bm`) or must not be played (`am`), grouped by theme. Currently 14/18.
Add a line whenever a rated game exposes a weakness. `kingsafety` and `endgame`
are the thinnest themes. The suite's wall-clock timing has real run-to-run
noise near a depth-completion boundary — before concluding a code change
caused a flip, replay the exact same `search_root(soft, hard)` call before and
after; if nodes/score/PV come back identical, it's noise, not a regression.

`analysis/tune.py` builds a variant of `agent.py` with exactly one line
rewritten, so an arena result attributes cleanly to one constant.

**The arena's default clock (10s+0.1s) is a different regime than rated
games.** A self-play game caps at depth 3-4 on that fast clock but reaches
depth 6-8 on the realistic clock (120s+0.5s), matching rated-game play. Use the
fast clock to screen, but **re-confirm any winner at `--base-ms 120000
--increment-ms 500`** before it ships — every pruning-constant sweep before
this was discovered was measured in the shallow regime only.

---

## What has been tried, and what it measured

### Shipped, with numbers

| Change | Result |
|---|---|
| v1.1 over v1 — board-unwind fix, fail-low extension, king safety | **73.3%** |
| v1.3 — repetition counting, low-clock budget cap | safety fixes, not separately measured |
| v1.4 over v1.3 — lazy eval + SEE at threshold −110 | **70.0%** |
| v1.6 over v1.5 — TT/check-extension depth fix, qsearch draw detection | safety fixes, sanity-checked only (12 games, 50.0%) |
| v1.7 over v1.6 — staged TT-move-first search | inconclusive: 43.8% over 40 games, within noise; kept, not reverted |
| v1.8 over v1.7 — numba-jitted material/PST/bishop pair | fast-clock arena was noise (51.2%/40 games, expected at that clock); kept on a bit-exact correctness proof instead |
| v1.9 over v1.8 — forced-reply extension, numba mobility/king-zone jit, incremental Zobrist | not arena-confirmed at all, given a tournament deadline; each verified to the strongest correctness standard available for its kind instead (bit-exact parity, debug-assert against `chess.polyglot.zobrist_hash`, or direct before/after comparison) |
| v20 over v1.9 (merged to main) — packed-int killers/counters/TT move slot + typed-array TT; numba-jitted SEE; incremental material/PST/phase eval | not arena-confirmed (see below); each phase verified to a correctness standard instead |

`submission.zip` is rebuilt from the current tree (main), not from a version
proven to beat the ladder — none of v1.7-v20 have arena confirmation, only
correctness verification. Rebuild it (`uv run python -m harness.package`) and
verify it runs standalone from a clean extract before treating it as ready to
upload.

**v20 (the former `hybrid-engine-state` branch, now merged), in more detail.**
Built at the user's
explicit request to override "Not doing further"'s rejection (below) of a
hybrid engine-state architecture, given a live tournament deadline. Three
phases landed, each independently verified rather than arena-tested:

- **Packed-integer moves + typed-array TT** (Open Work #9 below, now done):
  `KILLERS`/`COUNTER`/the TT's move slot hold a packed 15-bit int instead of a
  `chess.Move` object; `TT` is five parallel numpy arrays instead of a list of
  tuples. Verified with a fixed-depth parity harness (not committed): 20
  positions at depth 5 from an empty TT, identical move/score/nodes/qnodes
  before and after — a pure storage-layout change, node visitation unaffected.
- **Numba-jitted SEE**: `see()` used `board.attackers_mask()`, not
  numba-compatible; now jitted using the mobility jit's own ray-casting
  helpers plus new `PAWN_ATTACKS_NP`/`KING_ATTACKS_NP` tables (from
  python-chess's own `BB_PAWN_ATTACKS`/`BB_KING_ATTACKS`). Verified against
  the pure-Python version it replaced across 91,683 legal captures (123 en
  passant, 568 promotion) from the scenario suite plus 400 random-walk games:
  0 mismatches.
- **Incremental material/PST/phase evaluation**: `evaluate()`'s pass 1 no
  longer recomputes from scratch on every call when an incremental
  `mg_eg_phase` value is threaded through negamax/qsearch/`_root_window` (a
  structural twin of the incremental-Zobrist machinery, extended with
  material/phase/bishop-pair on top of the same piece-square delta shape).
  `EVAL_ASSERT` (mirrors `ZOBRIST_ASSERT`) caught a real bug on the first real
  test: the en passant branch used the mover's sign instead of the captured
  pawn owner's sign. Fixed, then re-verified clean: a depth-3 exhaustive
  sweep over 10 positions chosen for castling/promotion/bishop-pair-threshold
  coverage, 800 random-walk games checked at every ply, the scenario suite at
  depth 6, and 4 complete self-play games through checkmate/threefold/
  promotion. Since the incremental value is proven *exactly* equal to a fresh
  recompute rather than merely close, this is treated as the same class of
  change as the material/PST and mobility jits (provably unable to alter node
  visitation), not a behavior change needing the arena.

A 10-game spot-check against `philbert-v19` (+4 =4 −2, 60%) found no crashes,
illegal moves, or blunder pattern — not a strength claim at that sample size,
just a sanity check. **Extending staged move generation to captures** (the
essay's other recommendation, and the reason a "Phase D" was planned) was
deliberately not attempted: unlike the three phases above, it changes *which*
nodes get visited and cached — the same risk class the staged TT-move-first
search (`v1.7`) turned out to belong to despite looking behavior-preserving —
so a correctness proof cannot substitute for it, and it was not implemented
without a real 30-40 game arena to back it.

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
suggested, because the cheap pass still walks every piece. Superseded: material
and PST are now numba-jitted from scratch rather than incrementally maintained
in pure Python — comparable speedup, much lower correctness risk than hand-kept
incremental state through push/pop across captures, castling, en passant and
promotion.

**Staged TT-move-first search looked purely behavior-preserving and wasn't.** A
fixed-depth-6 parity check from an empty TT found 3 of 10 random positions
scored differently (12-76cp, no blunders) from the pre-change code. Not a bug:
skipping full move generation after an early TT-move cutoff means some
sub-nodes are never visited and never cached, and LMR/NMP/futility are
approximate enough that a different set of cached moves elsewhere can shift a
cold-TT score. Shipped anyway (43.8%/40 games, within noise, no correctness
defects) — but the lesson stands: "this can only change speed" needs checking,
not assuming, for anything that changes *which* nodes get visited, even without
touching a pruning decision directly.

### Pruning constant sweep, 30-40 games each

| Variant | vs | Score | Read |
|---|---|---|---|
| `rfp-40` (`RFP_MARGIN = 40`) | v1.4 | 61.7% | promising, only 1.5 SE — did not replicate (below) |
| `rfp-120` | v1.4 | 20.0% | clearly much worse, 3.9 SE below even |
| `rfp-40` re-test | v1.7 | 52.5% | within noise; the earlier 61.7% did not survive a cleaner baseline |
| `lmp-tight` | v1.4 | 55.0% | no information |
| `lmp-loose` | v1.4 | 48.3% | null |
| `lmr-aggr` / `lmr-shy` | v1.4 | 50.0% / 48.3% | null |
| `see-60` / `see-200` | v1.4 | 48.3% / 43.3% | both worse, confirms -110 is near optimal |

`RFP_MARGIN` stays at 75. The one apparently-sensitive result found did not
replicate against a cleaner baseline — a reminder that a result against a stale
baseline is not evidence against a current one. LMR and LMP are insensitive,
which together with an effective branching factor of 1.1-4.5 says pruning is
otherwise close to exhausted. `delta-100`, `delta-350`, `fut-tight` and
`fut-loose` were built but never arena-run; not pursued further this session
given the effect sizes involved are likely below the measurement floor anyway
(see "Not doing further" below).

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

### Bugs found by structural self-audit, not a rated-game loss (v1.6)

A pass tracing every pruning/ordering mechanism's shared state, done before
more tuning rather than after another loss:

- The TT probe compared a cached entry's depth against the pre-extension depth
  while the store used the post-extension depth, so an in-check node could
  accept a cutoff from an entry one ply shallower than the check extension
  exists to guarantee — at depth 0 specifically, almost any cached entry
  satisfied the requirement, defeating the extension outright. Fixed by
  computing the extension before the probe.
- `qsearch` had no draw detection. Verified with a hand-built stalemate
  (`k7/8/1Q6/8/8/8/8/7K b - - 0 1`): scored −1009 before the fix, 0 after.
- Futility pruning's gate is a strict superset of LMP's with no move-index
  limit, so wherever both are enabled `LMP_COUNT` cannot fire — a likely
  explanation for the `lmp-tight`/`lmp-loose` null result above. Not yet
  isolated; would need a futility-disabled diagnostic arena to confirm.

### Where the strength actually goes

- **Round 46** was lost from +161. One move, `Bc5`, scored +166 at depth 4, +94
  at depth 8 and −195 at depth 9. The engine reached depth 7. The evaluation was
  right; it could not see far enough. **After SEE, that position is solved
  correctly at depth 6** and it is a regression case in the suite.
- **Round 49** contained no move losing 150cp. It lost through a run of 50-110cp
  second-best moves. That is the profile of being outplayed, not of a bug.
- **Round 62** (v1.8, still current as of v1.9): `f5` looks merely bad through a
  complete depth-7 search (-781) but walks into a forced mate a fresh depth-7
  search of the position *after* `f5` finds immediately (-29991, via
  `analysis.blunders` scoring played vs best the same way). `Rd3+` holds a
  clearly losing but non-forced position (-1083) instead. Forcing `search_root`
  a full uninterrupted 30s directly (far more than any realistic per-move
  budget) does find the mate, but only partway into depth 8, timing out
  mid-search on `f5` itself before ever reaching an alternative to compare it
  against. Same failure family as round 46 (missed below depth 9). The
  forced-reply extension measurably deepens the critical junction in this exact
  line (after `Nxf5+` the king has exactly one legal reply, seldepth 14->21)
  but is not enough alone to flip the result — the actual refutation needs more
  than one extra ply there. Regression case in the suite.
- **Round 68** was lost by checkmate from a roughly balanced position out of a
  sound piece sac (Bxg6/Qxg6 around move 19, confirmed only 18cp off the
  alternative). A long run of 20-90cp second-best rook-endgame moves (round
  49's profile), then two moves confirmed as real losses (move 39 `Kd2`,
  -188cp against `Ke4`; move 43 `Kb2`, -176cp against `Kb4`). **Initially read
  as a king-activity/evaluation problem; a term-by-term eval breakdown of
  both positions overturned that.** The static evaluation right after each
  move is nearly tied (Kd2 -300 vs Ke4 -278; Kb2 -130 vs Kb4 -128 — low tens
  of cp apart, not the 176-188cp the search eventually finds). The real gap
  is a forced sequence invisible at that depth: a rook infiltration via
  checks after `Kd2` (PV `Rg3 Ke2 Rf5 e6 Rxg2+ Ke3 Rg3+ Ke4`), a
  tempo-gaining `...c3+` pawn race after `Kb2`. **Same failure family as
  round 46/round 62 (a forced sequence a few plies past where the search
  looked), not a distinct endgame-evaluation weakness** — both regression
  cases stay filed as "regression," not "endgame," for that reason. The
  already-open recapture/threat-extension item (Open Work below) targets
  this failure mode directly and is now additionally motivated by round 68,
  not just 46/62.

`analysis/blunders.py` was rebuilt this session around round 68: the old
version scored *every* legal move at *every* our-move with an independent
equal-depth child search, so a full game took hours, not minutes. It now
runs in two stages — a cheap fixed-depth pass over every legal move sharing
one TT/killers/history state (mirroring `_root_window`'s own algorithm
without touching that function), flagging a position as suspicious only on a
disagreement with the played move or a large eval swing; then, for
suspicious positions only, the original proven-sound equal-depth comparison
on a handful of candidates instead of all of them, with state now reset
*per candidate* rather than once per position (closing a fairness gap the
old version had). A full round-68 replay dropped from "still running after
20+ minutes, no visible progress" to under 2 minutes for the informative
part of the game. One real limitation stated plainly in its own docstring: a
shallow first pass cannot see a problem that only resolves below its depth,
so a genuine round-46/round-62-style horizon blunder can pass through
undetected — `--force-deep` exists for when one is suspected on other
grounds. It also surfaced a live bug while being built: `TimeUp` from a deep
forced-reply/check-extension chain (a promotion-with-check sequence near
mate took over 170s at nominal depth 7) was uncaught and, worse, would have
left the board several plies deep afterward with no error at all — fixed by
unwinding to the pre-candidate stack depth in a `finally`, the same
invariant `_root_window` already protects in `agent.py` itself.

- **Round 72** was lost by checkmate as Black. A different profile from every
  prior loss: **nothing crossed the 150cp blunder threshold anywhere in the
  game.** The score sat near-equal through move 16, then move 17 (`Bg4`
  played, −114, vs `Bxe4`, +2 — a 116cp gap, just under threshold) never
  recovered, degrading further from move ~40 into a lost rook ending — round
  49's "being outplayed" profile, with a much clearer starting point than
  round 49 had. `Bg4`'s refutation is shallow (`Qxf4` two plies later,
  grabbing a hanging pawn), and `Bxe4`'s true value (+2, not the +423 a
  static snapshot after it suggests) is exactly the single-square exchange
  `see()` is built to resolve — unlike round 68, not a clean "needs many
  plies" case. The real game spent its slowest move of the entire game (7.2s)
  on this exact position and still chose wrong; the platform's log
  truncation destroyed the trace that would explain why, which motivated
  building `analysis/trace.py` (below).
- **Round 74** was lost by checkmate as White. Unlike round 72, a confirmed
  real blunder: move 36, `Kh1` scores −837 against `Kg1`'s −453 (384cp).
  Same signature as round 68 again — static eval right after each move is
  nearly tied (`Kh1` −352 vs `Kg1` −365, 13cp apart) — a fourth confirmed
  instance of the round-46/62/68 failure family (forced sequence past
  reached depth), not a fifth pattern. Specifically: `Kh1` invites `...e3+`,
  which forces `Kg1` anyway but from a worse position, because of what the
  inserted tempo let Black set up first (PV `e3+ Kg1 Nf6 Rxg7+ Kf8 Rf7+ Qxf7
  Qxh6+ Qg7+ Qxg7+`). A replay-faithful `analysis.trace` of this exact
  position hit 13 replay-state mismatches earlier in this sharp game
  (round 72's trace had 1) and found only a razor-thin −588 vs −592 gap at
  the deepest depth reached — this position's difficulty is real even for
  search itself, not just an artifact of `analysis.blunders`' reset-state
  methodology. Both rounds' regression positions are in the suite.

`analysis/trace.py` was added this session, built on an optional trace
callback threaded through `search_root`/`_root_window` in `agent.py` (`None`
at the real `get_move()` call site, verified inert with the usual
fixed-depth parity technique). Replays a real PGN from the start via real
`get_move()` calls — not `analysis.blunders`' deliberately reset TT/killers/
history/counters — so a traced position's search runs under the same
accumulated state and the same real clock (read from the PGN's own `%clk`
annotations) a tournament game actually had, then reports every root
candidate's score/nodes/time/completion/telemetry at every depth, not just
the eventual winner. Its own central limitation, surfaced on the first two
real uses: replaying via real wall-clock timing can diverge from what the
original game's search reached at an earlier move (round 72: 1 mismatch;
round 74: 13, in a much sharper game), so a replayed trace is informative,
not a certified reconstruction of what the original tournament run computed.

### Measured profile

Nodes per second was 9-11k before this session's jits; roughly 15-25k after,
position-dependent (mobility jitting helps piece-dense positions more than
sparse ones with long-range sliding pieces — see the numba paragraph below).
Depth 6-8 in a rated game, measured before the speedup; not yet re-measured at
the new NPS. Effective branching factor 3.5-4.5 early, lower deeper — below the
sqrt(35) that perfect ordering approaches, so **move ordering is close to its
ceiling**. Every debug line reports `ebf`, so rated games measure this for
free.

**Numba jitting is real but not a clean multiplier for mobility.** Material/PST
was a clean ~2x. Mobility/king-zone, verified bit-exact across 4012 positions,
measured 7-10% faster on a sparse king+rook endgame, a wash on a busy
middlegame, and *slower* than the pre-jit version on one open position with a
queen on unobstructed diagonals and files — a queen ray-casts both diagonally
and orthogonally, and simple ray-casting (chosen over magic bitboards for lower
implementation risk) has no answer for "many empty squares to walk" the way a
lookup table does. Kept anyway, since bit-exactness means the only thing it can
affect is speed, never move choice at a given depth.

---

## Open work, in priority order

Anything here that claims a strength gain has to survive a 40 game arena
against the current frozen baseline before it ships (or, for a change provably
unable to alter node visitation — see "How changes get decided" — a bit-exact
or debug-assert-verified correctness proof can substitute). Screen at the
arena's default fast clock; **re-confirm any winner at realistic clock**
(`--base-ms 120000 --increment-ms 500`) before it ships.

1. **Recapture and threat extensions.** Cheap, and aimed at the round 46/62
   failure mode more directly than the forced-reply extension already shipped
   (which helps only when a node narrows to exactly one legal reply). Round 68
   moves 39/43 and round 74 move 36 (see above) are now a fourth confirmed
   instance of this exact failure mode, not just two — every one of them a
   forced sequence the search only resolves several plies past where a
   static-eval comparison would call the position a near-tie. This is now
   the single most repeated finding in the postmortem history and should
   move up from "priority 1 on a list" to "actually build it next."
2. **Measure whether the forced-reply extension is a net positive** across
   ordinary games, not just the two adversarial cases that motivated it. It's
   verified sound and triggers correctly; strength effect is genuinely unknown.
3. **A smarter sliding-attack representation for the mobility jit** (magic
   bitboards, or the o^(o-2r) trick) — only worth it if the open-position
   slowdown documented above turns out to matter in practice; simple
   ray-casting was the deliberate lower-risk choice the first time.
4. **Singular extensions.** Present in the 5x5 engine, never ported. Contested
   — they may only pay from depth 10. One arena settles it.
5. **Unconfound LMP from futility.** Build a diagnostic-only variant with
   `FUTILITY_MARGIN` set unreachably high, then sweep `LMP_COUNT` tight/loose
   against *that*, never against a real baseline. If a real gradient appears
   where none did before, `LMP_COUNT` has genuine signal that was masked; if
   still null, the existing conclusion stands on firmer ground. Never ship the
   futility-disabled variant itself.
6. **Finish the interrupted delta/futility sweep** (`delta-100`, `delta-350`,
   `fut-tight`, `fut-loose` — built via `analysis.tune --all`, never arena-run).
   Low expected value: `DELTA_MARGIN` and `QS_SEE_THRESHOLD=-110` prune
   overlapping bad-capture populations, so a null result wouldn't mean delta
   pruning is inert, just masked the same way LMP was.
7. **Sweep NMP's `R` formula** (`reduction = 2 + depth // 6`, never tuned).
   Same shadowing caveat as above: RFP returns before NMP is even reached under
   the same guard, so a null result may mean RFP is shadowing NMP, not that `R`
   doesn't matter.
8. **Fix and re-enable aspiration windows.** Two defects first: the re-search
   loop never checks the clock, and `search_root` accepts a fail-low's move
   even though it's only an upper bound.
9. ~~TT as typed arrays~~ — done, see the v20 entry above.
10. **Extend staged move generation to captures** (`generate_legal_captures()`
    searched before the full legal-move list is ever built, past the existing
    TT-move-first stage). Sketched out during the v20 work but not
    implemented: unlike v20's other three phases, this changes *which* nodes
    get visited and cached — the same risk class `v1.7`'s staged TT-move-first
    search turned out to belong to — so it needs a real 40-game arena, not a
    correctness proof, before it ships.
11. **Arena-confirm v20's history aging** (`HISTORY_AGING_ENABLED`, disabled
    by default in the shipped code): a bounded gravity update + malus for
    HISTORY, built and verified inert when off, never arena-tested. A spot
    check was started and interrupted by a rated-game postmortem (round 68)
    and not resumed.

**Not doing further, and why:**

- **ProbCut.** Forward pruning makes the search see less, so it cannot fix a
  horizon problem — the opposite of what rounds 46 and 62 need.
- **NNUE-style evaluation.** A more expensive eval at this nps costs the ply it
  is meant to buy.
- **History and move-ordering rework**, and **more RFP/LMR/NMP constant
  sweeps** beyond what's listed above. The measured branching factor says there
  is little headroom left in ordering/pruning generally; every sweep this
  project has run backs that up.
- ~~A parallel hand-rolled engine state (integer moves, incrementally
  maintained material/PST, a shadow board synced with `chess.Board`) to get
  past `python-chess`'s own overhead.~~ This rejection stood on its own
  merits (comparable theoretical payoff to numba, much higher correctness
  risk: state that must stay synchronized across every move type — captures,
  castling, en passant, promotion), but was overridden at the user's explicit
  request despite the live tournament clock; see the v20 entry above for what
  shipped and how it was verified instead of arena-tested. `chess.Board`
  stayed the sole legality/correctness oracle throughout — nothing here
  reimplements movegen or legality.

**Ongoing, not a single item:**

- **Grow the scenario suite.** `kingsafety` and `endgame` are the thinnest
  themes. Add a line to `analysis/positions.epd` after every rated game that
  shows a weakness. One suite expectation is still unverified and may be wrong
  rather than the engine: `take the opposition` prefers `e4` over `Ke2`, and
  needs checking against a real endgame reference.
- **Freeze a new baseline whenever a version ships**, so the next change is
  measured against what it would actually replace.
- **Read the per-game stderr log after every rated loss.** Every engine bug
  found so far was found that way, not by local play.

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

`docs/ENGINE_BRIEF.md` is a standalone version of this file's architecture and
findings sections, written for outside review. Keep it in sync when either
changes materially.
