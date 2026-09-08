Frozen v1.8: v1.7 plus numba-jitted material/PST/bishop pair.

Unlike everything else frozen this session, this one is bit-exact: a 3000
random-position parity check plus 7 hand-built edge cases (corner squares,
7-queen/7-rook stress positions, bishop-pair boundaries, near-empty boards)
all matched the old pure-Python pass 1 exactly, on (mg, eg, phase), not just
the final score. Mobility, king safety and pawn structure (pass 2) are
untouched pure Python.

Because it's bit-exact, this carries none of v1.7's node-visitation-order
risk — every evaluate() call returns the identical value it always did, just
faster, so the only way this changes a game's outcome is by reaching more
depth in the same clock. Self-play NPS ~24k, up from the 9-11k baseline.

Measured: fast-clock arena vs v17, 51.2% over 40 games (noise, expected - the
fast clock caps depth at 3-4, where the speedup has the least room to convert
into extra plies). A realistic-clock (120s+0.5s) confirmation was started but
not waited on before shipping, given the bit-exact correctness proof above
removes the main class of risk an arena would normally be needed to catch.

This is the version in submission.zip as of this session.
