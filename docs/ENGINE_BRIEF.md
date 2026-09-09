# Chessathon engine: technical brief

A self-contained description of our chess engine, its measured behaviour, and
the open questions. Written to be handed to someone (or something) that has not
seen the code, so it can give specific advice rather than generic advice.

Last updated 2026-09-08.

---

## 1. The constraints

These are fixed by the competition and are not negotiable.

| | |
|---|---|
| Language | Python 3.12 |
| Available packages | stdlib, `python-chess 1.11.2`, `numpy 2.5.2`, `numba 0.67.0`, `torch 2.13.0+cpu`, `onnxruntime 1.29.0` — **nothing else installs** |
| CPU | one core of an AMD EPYC 9V74 at 2.60 GHz |
| Memory | 2 GB |
| Threads | more than one costs time; there is one core |
| Network | none, in either direction |
| Time control | 120 s per side, plus 0.5 s per move |
| Init budget | 90 s at import, before the clock starts |
| Submission | a zip, `agent.py` at the root, 50 MB unzipped max, no native binaries |
| Entry point | `def get_move(fen: str, time_left_ms: int) -> str` returning UCI |
| Game end | FIDE rules; 600 plies is a draw |
| Failure modes | illegal move, crash, or flag all lose outright |

Third-party engines are banned, including any port or wrapper. A database of
another engine's moves shipped for runtime lookup counts as an engine. Any
neural network must be trained by us; starting from a published chess network is
not allowed. Training on positions labelled by an existing engine **is** allowed.

Rated games start from curated near-equal opening positions, not the standard
start, so an opening book keyed on move one is largely wasted.

---

## 2. What the engine is now

Pure Python on top of `python-chess` for move generation and legality. We do not
hand-roll bitboards: an illegal move is an instant loss, and `python-chess` is
correct. Where speed matters, the code reads `python-chess`'s own raw bitboard
attributes directly (`board.pawns`, `board.occupied_co[color]`, etc.) rather
than going through slower object-returning APIs (`piece_at`, `pieces`).

**Search** — negamax with alpha-beta, principal variation search, iterative
deepening under a soft/hard time split that unwinds through an exception.

**Already implemented**

- Transposition table, 2^20 buckets, exact/lower/upper flags, ply-adjusted mate
  scores, replacement on depth
- Incremental Zobrist hashing in the hot recursive call sites (negamax's staged
  TT-move branch, its fallback move loop, and the root search), threaded as an
  optional argument that falls back to a full `chess.polyglot.zobrist_hash`
  recompute when not supplied — no call site can produce a wrong key, only a
  slower-but-correct one if it isn't wired up
- Staged move generation: the TT's cached move, when legal, is tried via the
  cheap `board.is_legal()` check before the full legal-move list is ever
  generated; only a miss pays for full generation and ordering
- Quiescence search with delta pruning and SEE-based cutoffs, plus its own
  draw/stalemate detection (50-move rule, insufficient material, threefold
  repetition on the in-check branch; a material-gated stalemate check on the
  non-check branch)
- Null move pruning, `R = 2 + depth/6`
- Late move reductions, `int(0.75 + log(d) * log(i) / 2.25)`
- Late move pruning, futility, reverse futility
- Check extensions, and a forced-reply extension (one legal move costs nothing
  extra to search one ply deeper)
- Internal iterative deepening
- Move ordering: TT move, then MVV-LVA captures gated by SEE, promotions,
  killers, counter-moves, history
- Static exchange evaluation, validated against a brute-force reference
- Lazy evaluation, skipping the positional pass when material and PST already
  sit 600 cp outside the caller's window

**Evaluation** — tapered midgame/endgame throughout, blended on a phase count of
24. Material, piece-square tables and the bishop pair are numba-jitted (bit-exact
against the pure-Python version). Mobility and the king-zone-attack term
(knights/bishops/rooks/queens) are also numba-jitted, via hand-rolled ray-casting
for sliding pieces since `python-chess`'s own sliding-attack tables are Python
dicts numba can't index efficiently. Pawn structure, rook-file checks, and king
shield/open-file checks stay pure Python — simple bitwise-mask operations with
no attack generation, so jitting them doesn't pay off the same way.

**Not implemented** — singular extensions, ProbCut/multi-cut, incremental
material/PST maintenance in pure Python (superseded by jitting the full
recompute instead), any neural network.

**Disabled** — aspiration windows: implemented, and measurably worse (see §4).

---

## 3. Measured behaviour

Everything here is measured on the development machine, not guessed.

| Metric | Value |
|---|---|
| Nodes per second, pre-numba baseline | 9,000–10,000 |
| Nodes per second, after material/PST + mobility jit + incremental Zobrist | roughly 15,000–25,000, position-dependent (see §4) |
| Depth reached in a real game | 6–8 (measured before this session's speedups; not yet re-measured at the new NPS) |
| Effective branching factor, plies 2–4 | 3.5–4.5 |
| Effective branching factor, plies 5–7 | 1.1–1.5 |
| Share of runtime in `evaluate`, pre-jit | ~42% |
| Share of runtime in `zobrist_hash`, pre-incremental | ~8% |
| Share of nodes inside quiescence | ~66% |

Perfect move ordering in chess approaches sqrt(35), about 6. We are below that,
which says **move ordering is close to its ceiling**. Nodes per second was the
identified remaining lever, and is where this session's work went; the EBF
figures above predate the speedups and should be re-measured once a stable
baseline reflects them.

### Version history

Measured over 30–40 games at 10 s + 0.1 s, alternating colours, except where
noted. **A methodology gap found this session**: this fast clock caps search
depth at 3–4 in this engine, while a realistic clock (120 s + 0.5 s) reaches
6–8, matching real rated games. Every entry through v1.4 predates this finding;
newer entries are noted where a realistic-clock re-confirmation was or wasn't
done.

| Change | Result |
|---|---|
| v1.1 over v1 — board-corruption fix, fail-low extension, king safety | **73.3%** |
| v1.4 over v1.3 — lazy eval + SEE at the tuned threshold | **70.0%** |
| v1.6 over v1.5 — TT/check-extension depth fix, qsearch draw detection | safety fixes, sanity-checked only (12 games, 50.0%) |
| v1.7 over v1.6 — staged TT-move-first search | inconclusive: 43.8% over 40 games, within noise; kept, not reverted (see §4) |
| v1.8 over v1.7 — numba-jitted material/PST/bishop pair | fast-clock arena was noise (51.2%/40 games, expected at that clock); kept on the strength of a bit-exact correctness proof instead |
| v1.9 over v1.8 — forced-reply extension, numba mobility/king-zone jit, incremental Zobrist | not arena-confirmed at all, individually or combined, given a tournament deadline; each verified to the strongest correctness standard available for its kind (see §4) |
| v20 over v1.9 (merged to main) — packed-int killers/counters/TT move slot + typed-array TT; numba-jitted SEE; incremental material/PST/phase eval | not arena-confirmed; each phase verified to a correctness standard instead (see §4). Built at explicit user request, overriding an earlier "not worth the risk" rejection of this architecture. |
| Recapture extension over v20 (+1 ply on the opponent's last-move square) | **lost the arena, 40.0% (+8 =8 −14, 30 games)** — reverted; see §4 |
| Aspiration windows | −15.6% nodes to depth 8; **lost the arena, 42.5%, with one game lost by flagging** — rejected, still disabled |

`analysis/trace.py` (new this session) replays a real PGN via real `get_move()`
calls, so a traced position runs under the same accumulated TT/killers/history
and the same real clock the tournament game had, then reports every root
candidate's score/nodes/completion at every depth — built specifically to
answer why the recapture extension's target failure mode (rounds 46/62/68/74:
static eval nearly tied, real gap only several plies deeper) kept recurring.
Traced properly rather than left as a hypothesis, those four losses turned out
to share no single mechanism: rounds 46, 62 and 74 look like convergence/depth
sensitivity, not a structural miss; rounds 68/72, replayed under real
accumulated state, actually find the *correct* move — directly weakening
"ordering is broken" as a general explanation. Round 62's `f5` line was dug
into further in a later session: the old read (`f5`'s subtree alone exhausts
a depth iteration's time budget, so `search_root` falls back to a stale
answer) turned out to be an artifact of tracing from a cold state on the old,
slower engine. Replaying the real game's accumulated state and real clock on
the current engine shows no timeout at all — depth 9 completes in well under
a second and returns `f5` (matching what was played) because `f5` and `Rd3+`
score within 6cp of each other there. The expensive blowup is real (a
cold-state trace with a large time budget reproduces it and finds `f5`'s true
score is a forced mate) but warm ordering only delays it, past any realistic
per-move budget; a ply-by-ply walk of the actual mating line shows check and
forced-reply extensions earning it +7 plies over a nominal 10-ply length,
needing effective depth ~17 to resolve. Both extensions fire correctly — this
is a genuine horizon/time-budget shortfall under real clock pressure, not a
bug, and is now a data point for trying singular extensions (§5 open
questions) rather than its own open item. See CLAUDE.md's "How changes get
decided" for the full evidence table.

---

## 4. Negative results, and near-misses that looked like negative results

**Lazy evaluation returned 5–8%, not the large win the 42% profile share
suggested.** Splitting `evaluate` into a cheap pass and a positional pass still
walked every piece in the cheap pass, so the split cost most of what the cut
saved. Superseded: material/PST is now numba-jitted rather than incrementally
maintained in pure Python.

**SEE halved the node count and won nothing — until one constant changed.**
Quiescence prunes captures whose SEE falls below a threshold:

| Threshold | Score against the same opponent |
|---|---|
| 0 — prune every capture that loses any material (textbook) | 48.8% |
| −100000 — no pruning, ordering only | 45.0% |
| **−110 — prune only captures losing more than a pawn** | **76.2%** |

Pruning every SEE-negative capture discards the exchanges the search needs to
resolve, and mis-scored ones too, since SEE ignores pins. This was a 26 point
swing from a single number, and remains the largest, clearest strength result on
record in this project.

**A "free" speed optimization turned out not to be free.** Staged TT-move-first
search (skip full move generation when the cached TT move alone produces a
cutoff) looked purely behavior-preserving. A fixed-depth-6 parity check from an
empty TT found 3 of 10 random positions scored differently (12–76 cp, no
blunders) from the pre-change code. Not a bug: skipping generation after an
early cutoff means some sub-nodes are never visited and never cached, and
LMR/NMP/futility are approximate enough that a different set of cached moves
elsewhere can shift a cold-TT score. A 40-game arena came back within noise
(43.8%). Lesson: "this can only change speed" needs to be checked, not assumed,
for any change that alters *which nodes get visited*, even if it changes no
pruning decision directly. (Numba jitting, by contrast, genuinely is
speed-only — it doesn't change which nodes get visited, only how fast each one
evaluates, which is why it was shipped on a bit-exact proof rather than an
arena result.)

**Numba mobility jitting is a real but mixed speedup, not a clean multiplier.**
Direct paired comparisons on the same positions: a sparse king+rook endgame ran
7–10% faster; a busy, fully-developed middlegame was a wash; one open position
with a queen on unobstructed diagonals and files ran slower than the
material/PST-only baseline. A queen ray-casts both diagonally and
orthogonally — double the work of a rook — and simple ray-casting (chosen over
magic bitboards for lower implementation risk) has no answer for "many empty
squares to walk" the way a lookup table does. Kept anyway: proven bit-exact
against the pure-Python version across 4012 positions, so the only thing it can
affect is speed, never move choice at a given depth.

**An assert-against-ground-truth debug flag caught a real bug on its first real
test.** Incremental material/PST/phase evaluation (mirroring the incremental
Zobrist key's shape: a delta computed before a push, threaded through
negamax/qsearch as an optional argument, every non-threading call site still
correct just slower) shipped with `EVAL_ASSERT`, a flag that recomputes pass 1
from scratch and asserts it matches the incremental value on every call —
exactly the pattern `ZOBRIST_ASSERT` already used. The very first assertion
trip found a real bug: the en passant branch used the mover's sign instead of
the captured pawn owner's sign, silently writing every en passant capture's
material/PST swing in the wrong direction. Fixed, then re-verified clean
across 800 random-walk games checked at every ply, a depth-3 exhaustive sweep
over positions chosen for castling/promotion/bishop-pair coverage, and 4 full
self-play games through checkmate and promotion. Because the incremental
value is now proven *exactly* equal to a fresh recompute, not merely close,
this was treated as the same class of change as the material/PST and
mobility jits — provably unable to alter node visitation — rather than one
needing an arena.

**Two confident diagnoses from reading game logs were wrong when measured.** A
king walking to g4 in a rook ending looked like a king-safety bug; measured, it
was correct technique, because king danger is a midgame term that phase tapers
away by design. Halving the node count with SEE looked like it must help; it did
not, until the threshold was fixed.

**A tuning sweep's first result was a perfect null**, 50.0% over 30 games. With
±7–8% standard error at n=40, only large effects are detectable at all.
`RFP_MARGIN=40`'s apparently-promising 61.7% against an old baseline did not
replicate (52.5% against a cleaner one) once re-tested — a reminder that a
result against a stale baseline is not evidence against a current one.

**Two real bugs were found by a structural self-audit, not a rated-game loss.**
The TT probe compared a cached entry's depth against the *pre*-check-extension
depth while the store used the *post*-extension depth, letting an in-check node
accept a cutoff from an entry shallower than the extension exists to guarantee.
And `qsearch` had no draw detection at all — verified with a hand-built
stalemate FEN scoring −1009 before the fix, 0 after. Both shipped as v1.6.

---

## 5. How failures get diagnosed

Two tools, both in `analysis/`.

`blunders.py` replays a rated PGN and, at each of our moves, scores both the
move played and the best move **the same way** — push it, search the child to a
fixed depth, negate. Comparing two child searches at equal depth is sound;
comparing a child search against a parent search is not, and produced 300 cp of
phantom loss on moves that were in fact best.

`suite.py` runs `positions.epd`, where each line is a FEN plus a move the engine
must find or must not play, grouped by theme. Currently 17/22 at the suite's
default 3-second-per-position budget (that number itself has some inherent
run-to-run timing noise near depth-completion boundaries, confirmed this
session by replaying an identical `search_root` call twice and getting
byte-identical results — the suite's wall-clock methodology is more sensitive
to that than it first appears). Every rated game that exposes a weakness adds a
line.

### What the tools found in three losses (plus three more since)

- **Round 46** — one deep tactical miss. The losing move scored +166 at depth 4,
  +94 at depth 8, and −195 at depth 9. The engine reached depth 7 in the game.
  The evaluation was right; it could not see far enough. *After SEE, the same
  position is now solved correctly at depth 6.*
- **Round 49** — no move lost 150 cp. It lost through a run of 50–110 cp
  second-best moves, from +37 at move 44 to −405 at move 54. That is the profile
  of being outplayed, not of a bug.
- **Round 62** (v1.8, still current as of v1.9) — `f5` scored −781 through a
  *complete* depth-7 search from the root, but a fresh depth-7 search of the
  position immediately *after* `f5` finds a forced mate (−29991) the shallower
  combined search never sees. `Rd3+` holds a clearly losing but non-forced
  position (−1083) instead. Confirmed still current, not fixed by anything this
  session shipped: the live engine still plays `f5` here under realistic time
  budgets. Same failure family as round 46 — the evaluation is not the
  problem, search depth on a forcing sequence is. The forced-reply extension
  (added this session) measurably deepens the critical junction in this exact
  line (after `Nxf5+`, the king has exactly one legal reply) but is not enough
  on its own to flip the result — the actual refutation needs more than one
  extra ply there.
- **Rounds 76-78**, three further rated losses, none needing a code fix (no
  illegal moves, no crashes, no flag losses in any of the three). 76 and 78
  are mostly the round-49 "being outplayed" profile again, each with one or
  two moves confirmed as real >150cp errors late in an already-bad position.
  Round 77 is the clean case: Black was +330 to +450 for most of the second
  half of the game, then lost to a move (`f4`) that scored positive at every
  depth a normal shallow scan or the real game's own clock-limited search
  ever reached, but is a confirmed forced mate three plies further in, with a
  much better move sitting right there and not taken. Same family as round
  46/62 — search depth on a forcing sequence, not evaluation — and the
  cleanest isolation of it so far, since both moves were re-verified with
  identical reset state at the same depth.

---

## 6. The open questions

Specific answers are more useful than a survey. Several questions from the
previous version of this brief are now answered by this session's work; this
list reflects what's actually still open.

**On speed**

1. Mobility/king-zone jitting is done but mixed (§4) — real in piece-dense
   positions, a wash or worse where a piece has very long, unobstructed lines
   of sight. Is a smarter sliding-attack representation (magic bitboards, or
   the o^(o−2r) trick) worth the implementation risk for the positions where
   ray-casting loses, or is the current result good enough given time
   constraints?
2. Local-variable binding in the hot loop (`board.push`/`pop`/`is_check`
   bound to locals at the top of `negamax`) was measured directly this
   session rather than assumed, and the result was too noisy at small node
   counts to draw a conclusion either way — worth a cleaner, larger-scale
   measurement if there's time.
3. What is the actual node-per-second ceiling reachable in this stack
   (`python-chess` for movegen/legality, numba for evaluation only) before the
   Python search loop itself becomes the dominant cost? Is there a
   meaningfully cheaper way to represent the search loop's own bookkeeping
   (TT as parallel arrays instead of a list of tuples, for instance) that
   doesn't require abandoning `python-chess`?

**On search**

4. The forced-reply extension is shipped but strength-unmeasured. It's
   verified to trigger correctly and to be sound (no correctness risk), but
   whether it's a net positive across ordinary games — not just the two
   adversarial cases that motivated it — is still an open question.
5. Singular extensions remain unimplemented. At depth 6–8, is the node cost
   worth it, or do they only pay from depth 10 upward? Recapture/threat
   extensions, a smaller and cheaper idea in the same family, were tried and
   reverted (lost the arena, 40.0% — see §4); round 62's traced `f5` line
   (see §3) is a quantified data point for singular extensions specifically,
   since it needs effective depth ~17 to resolve, well past what recapture/
   threat extensions' single extra ply could reach.
6. What is the highest-value technique we have not listed at all?

**On machine learning**

7. Our reasoning is that an NNUE-style evaluation makes this engine's current
   nps *slower* and costs the ply it is meant to buy, and that the only ML
   shape which helps a shallow searcher is a small move-ordering policy net,
   because better ordering shrinks the tree rather than taxing it. Is that
   reasoning correct at this scale? If a policy net is right, what size and
   what input encoding is realistic to train and run in pure Python or ONNX on
   one core?

**On evaluation**

8. Our king-safety and endgame test coverage is thin. Which specific positional
   understandings most reliably separate a ~1600 engine from a ~2000 one at
   depth 7, now that material/PST/mobility evaluation is fast enough that
   adding a term costs less than it used to?
