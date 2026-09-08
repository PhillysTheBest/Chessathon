Frozen v1.7: v1.6 plus staged TT-move-first search in negamax.

negamax always built and ordered the full legal-move list even when the TT
move alone (always searched first when present) would already cut off.
Board.is_legal() checks pseudo-legality plus a single into-check test without
generating the full move list, so trying the TT move through it first and
only generating the rest on a miss saves that generation cost whenever the TT
move alone resolves the node.

Not as free as it looked. A fixed-depth-6 parity check from an empty TT found
3 of 10 positions score differently (12-76cp, no sign flips) from v1.6 —
skipping generation after an early cutoff means some sub-nodes are never
visited, so they never get cached, and LMR/NMP/futility are approximate
enough that a different set of cached moves elsewhere can shift a cold-TT
score. The more representative check (real iterative deepening via the CLI,
warmed-up TT) returned identical scores and PVs across the same positions
plus the r46 and "Nc2 drifts" regression FENs.

Measured: a 12-game fast-clock smoke arena against v16 came back 45.8%; a
40-game fast-clock arena came back 43.8% (+12 =11 -17). Both are within
CLAUDE.md's own noise floor (~7-8% SE at this sample size) — not the kind of
clear signal that shipped or killed anything else in this file. No crashes,
illegal moves, or anomalous terminations across either arena. Kept rather
than reverted: sound reasoning, no correctness defects, and an inconclusive
result isn't evidence of a real regression. Revisit if a future arena result
looks different against this baseline.
