Frozen v1.6: two correctness fixes over v1.5, found by a self-audit rather than
a rated-game loss.

- TT probe/store depth mismatch around the check extension. The probe used to
  compare a cached entry's depth against the pre-extension depth while the
  store used the post-extension depth, so an in-check node could accept a
  cutoff from an entry one ply shallower than the extension exists to
  guarantee. Fixed by computing the extension before the probe, so both sides
  compare the same depth.
- qsearch had no draw detection at all. Added a 50-move/insufficient-material/
  repetition check to the in-check branch (mirroring negamax's own guard), and
  a cheap, material-gated stalemate check to the non-check branch, so a
  quiescence leaf can no longer be misjudged as a live position when it is
  actually drawn.

Verification: ruff/mypy clean, self-test runs, analysis.suite shows two
scenario-suite items flipping at their default 3-second budget (r46's Bc5 case,
"Nc2 drifts") — investigated directly rather than taken as a regression. Given
a realistic clock (60s) the fix still reaches depth 6 on the r46 position in
2.4s and finds the correct move; the "Nc2 drifts" position drifts to Nc2 on
BOTH the old and new code at depth 8, so that flip predates this change.
analysis.blunders' played==best invariant holds (loss 0 in every case checked).

Strength: only sanity-checked, not confirmed. A 12-game fast-clock smoke test
against v15 came back +5 =2 -5 (50.0%) with no crashes, illegal moves, or
anomalous terminations (one draw by insufficient material, one by the 50-move
rule) — far too few games for a strength claim, just evidence nothing broke.
These are correctness fixes, not tuned constants, so per the v1.3 precedent
they are not separately arena-justified before shipping. A full 40-game arena
against v15 has not been run.
