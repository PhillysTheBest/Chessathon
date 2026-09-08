Frozen v20: v19 plus the hybrid-engine-state work, none arena-confirmed
individually given the time constraints this session ran under (a live
tournament), each verified by the strongest correctness method available
for its kind instead — the same tradeoff already made for v1.9's three
changes.

- **Packed-integer moves + typed-array TT.** KILLERS/COUNTER/the TT's move
  slot pack a chess.Move into a 15-bit int instead of holding the object;
  TT is five parallel numpy arrays instead of a list of tuples. Verified
  with a fixed-depth parity check: 20 positions, depth 5, empty TT,
  identical move/score/nodes/qnodes before and after - a pure storage-layout
  change, node visitation unaffected.
- **Numba-jitted SEE.** Reuses the mobility jit's own ray-casting helpers
  plus new pawn/king attack tables taken directly from python-chess.
  Verified against the pure-Python version it replaced across 91,683 legal
  captures (123 en passant, 568 promotion) from the scenario suite plus 400
  random-walk games: 0 mismatches.
- **Incremental material/PST/phase evaluation.** A structural twin of the
  incremental-Zobrist machinery, threaded through negamax/qsearch the same
  way. An EVAL_ASSERT debug flag (mirrors ZOBRIST_ASSERT) caught a real bug
  on its first real test - the en passant branch used the mover's sign
  instead of the captured pawn owner's sign - fixed, then re-verified clean
  across 800 random-walk games checked at every ply, a depth-3 exhaustive
  sweep over castling/promotion/bishop-pair edge cases, the scenario suite
  at depth 6, and 4 complete self-play games. Proven exactly equal to a
  fresh recompute, not merely close, so treated as the same class of change
  as the jits above rather than a behavior change needing an arena.
- **Move-ordering telemetry** (DEBUG-only, cutoff/NMP/LMR stats) and a
  verified-already-adequate numba warm-up check. Purely additive/diagnostic.

History aging (bounded gravity update + malus for HISTORY) is present but
disabled by default (HISTORY_AGING_ENABLED = False) - built and verified
inert when off, not arena-tested, not part of what this freeze represents
as active behavior.

`analysis/blunders.py` was also rebuilt this session (two-stage: a cheap
shared-state fixed-depth scan, then equal-depth verification only on
suspicious positions) - not part of agent.py, no effect on this baseline.
