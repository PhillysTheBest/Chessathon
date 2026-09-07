# Chessathon engine: technical brief

A self-contained description of our chess engine, its measured behaviour, and
the open questions. Written to be handed to someone (or something) that has not
seen the code, so it can give specific advice rather than generic advice.

Last updated 2026-09-07.

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
correct.

**Search** — negamax with alpha-beta, principal variation search, iterative
deepening under a soft/hard time split that unwinds through an exception.

**Already implemented**

- Transposition table, 2^20 buckets, exact/lower/upper flags, ply-adjusted mate
  scores, replacement on depth
- Quiescence search with delta pruning and SEE-based cutoffs
- Null move pruning, `R = 2 + depth/6`
- Late move reductions, `int(0.75 + log(d) * log(i) / 2.25)`
- Late move pruning, futility, reverse futility
- Check extensions
- Internal iterative deepening
- Move ordering: TT move, then MVV-LVA captures gated by SEE, promotions,
  killers, counter-moves, history
- Static exchange evaluation, validated against a brute-force reference
- Aspiration windows, +/-25 around the previous score from depth 5
- Lazy evaluation, skipping the positional pass when material and PST already
  sit 600 cp outside the caller's window

**Evaluation** — tapered midgame/endgame throughout, blended on a phase count of
24. Material, piece-square tables, mobility, bishop pair, doubled/isolated/passed
pawns, rook on open and half-open files, pawn shield, open files beside the king,
and a king-attack term on a quadratic danger curve.

**Not implemented** — singular extensions, ProbCut/multi-cut, incremental
evaluation, any numba, any neural network.

---

## 3. Measured behaviour

Everything here is measured on the development machine, not guessed.

| Metric | Value |
|---|---|
| Nodes per second | **9,000–10,000** |
| Depth reached in a real game | **6–8** |
| Effective branching factor, plies 2–4 | 3.5–4.5 |
| Effective branching factor, plies 5–7 | 1.1–1.5 |
| Share of runtime in `evaluate` | **~42%** |
| Share of runtime in `zobrist_hash` | ~8% |
| Share of nodes inside quiescence | ~66% |

Perfect move ordering in chess approaches sqrt(35), about 6. We are below that,
which says **move ordering is not the bottleneck**. Nodes per second is.

### Version history, each measured over 30–40 games at 10 s + 0.1 s, alternating colours

| Change | Result |
|---|---|
| v1.1 over v1 — board-corruption fix, fail-low extension, king safety | **73.3%** |
| v1.4 over v1.3 — lazy eval + SEE at the tuned threshold | **70.0%** |
| Aspiration windows | −15.6% nodes to depth 8; strength arena pending |

---

## 4. Negative results, which are the useful part

**Lazy evaluation returned 5–8%, not the large win the 42% profile share
suggested.** Splitting `evaluate` into a cheap pass and a positional pass still
walks every piece in the cheap pass, so the split costs most of what the cut
saves. Lazy evaluation appears to need an *incrementally maintained*
material/PST total to pay properly.

**SEE halved the node count and won nothing — until one constant changed.**
Quiescence prunes captures whose SEE falls below a threshold:

| Threshold | Score against the same opponent |
|---|---|
| 0 — prune every capture that loses any material (textbook) | 48.8% |
| −100000 — no pruning, ordering only | 45.0% |
| **−110 — prune only captures losing more than a pawn** | **76.2%** |

Pruning every SEE-negative capture discards the exchanges the search needs to
resolve, and mis-scored ones too, since SEE ignores pins. This was a 26 point
swing from a single number.

**Two confident diagnoses from reading game logs were both wrong when measured.**
A king walking to g4 in a rook ending looked like a king-safety bug; measured, it
was correct technique, because king danger is a midgame term that phase tapers
away by design. Halving the node count with SEE looked like it must help; it did
not, until the threshold was fixed.

**A tuning sweep's first result was a perfect null**, 50.0% over 30 games. With
±7% standard error at that sample size, only large effects are detectable at all.

---

## 5. How failures get diagnosed

Two tools, both in `analysis/`.

`blunders.py` replays a rated PGN and, at each of our moves, scores both the
move played and the best move **the same way** — push it, search the child to a
fixed depth, negate. Comparing two child searches at equal depth is sound;
comparing a child search against a parent search is not, and produced 300 cp of
phantom loss on moves that were in fact best.

`suite.py` runs `positions.epd`, where each line is a FEN plus a move the engine
must find or must not play, grouped by theme. Currently 12/14. Every rated game
that exposes a weakness adds a line.

### What the tools found in two losses

- **Round 46** — one deep tactical miss. The losing move scored +166 at depth 4,
  +94 at depth 8, and −195 at depth 9. The engine reached depth 7 in the game.
  The evaluation was right; it could not see far enough. *After SEE and
  aspiration windows, the same position is now solved correctly at depth 6.*
- **Round 49** — no move lost 150 cp. It lost through a run of 50–110 cp
  second-best moves, from +37 at move 44 to −405 at move 54. That is the profile
  of being outplayed, not of a bug.

---

## 6. The open questions

Specific answers are more useful than a survey. Assume the constraints in
section 1 are hard, that we have roughly three days, and that "measured over 40
games" is the standard of evidence.

**On speed, which is the identified bottleneck**

1. `evaluate` is 42% of runtime and cannot see `chess.Board` objects from numba.
   What is the least error-prone way to move it into numba — pass the eight
   piece bitboards plus occupancy as `uint64` scalars, pack into a numpy array,
   or something else? What realistically comes back at this size?
2. Is incremental material/PST through push/pop worth it in Python, given that
   the bookkeeping for castling, en passant and promotion is where the bugs
   live? Or does the interpreter overhead eat the gain?
3. `chess.polyglot.zobrist_hash` recomputes from scratch and is ~8% of runtime.
   Is an incremental Zobrist worth the risk of desynchronising from the board?

**On search**

4. We have LMR, LMP, futility, reverse futility and null-move pruning, and EBF is
   already 1.1–4.5. Is there meaningful headroom left in pruning, or is that
   exhausted and the remaining gains purely in nodes per second?
5. Singular extensions are the obvious next addition and directly target the
   round 46 failure. At depth 6–8, is the node cost worth it, or do they only pay
   from depth 10 upward?
6. What is the highest-value technique we have not listed at all?

**On machine learning**

7. Our reasoning is that an NNUE-style evaluation makes a 10k nps engine *slower*
   and costs the ply it is meant to buy, and that the only ML shape which helps a
   shallow searcher is a small move-ordering policy net, because better ordering
   shrinks the tree rather than taxing it. Is that reasoning correct at this
   scale? If a policy net is right, what size and what input encoding is
   realistic to train and run in pure Python or ONNX on one core?

**On evaluation**

8. Our king-safety and endgame test coverage is thin. Which specific positional
   understandings most reliably separate a ~1600 engine from a ~2000 one at
   depth 7, given that anything added costs time in the term that is already 42%
   of runtime?
