Frozen v1.9: v1.8 plus three changes, none arena-confirmed individually given
the time constraints this session ran under, each verified by the strongest
correctness method available for its kind instead.

- **Forced-reply extension.** One legal move extends the search one ply,
  since branching factor 1 costs nothing extra. Verified triggering where
  expected (round 62's Nxf5+/Kf7 junction, seldepth 14->21) but does not
  alone flip either round-46 or round-62's regression case — round 62 needs
  more than one extra ply at that junction, round 46's forcing line never
  reduces to a single legal reply at all. Sound and safe (no correctness
  risk), strength effect genuinely unmeasured.
- **Numba-jitted mobility and king-zone-attack evaluation.** Bit-exact
  against the pure-Python version across 4012 positions including 7-knight/
  7-rook/7-bishop/7-queen edge cases. Speed effect is real but mixed, not a
  clean further multiplier: faster in piece-dense middlegames and some
  sparse endgames, a wash or slightly slower where a queen has long
  unobstructed diagonals and files (ray-casting, chosen over magic bitboards
  for lower implementation risk, costs more per square of visibility than
  python-chess's own lookup does there). Kept because it's bit-exact - the
  only thing it can affect is speed, never move choice at a given depth.
- **Incremental Zobrist hashing** in negamax's hot call sites. Verified with
  a debug-assert flag against chess.polyglot.zobrist_hash across the full
  scenario suite, 9 hand-built edge cases (all 4 castling variants,
  promotion with/without capture, en passant, rights revoked by a rook
  moving and by a rook being captured), and a full 147-ply self-play game -
  zero assertion failures. 7-11% faster at every depth in a direct paired
  comparison, identical scores/nodes/PV confirming correctness under the
  speed change.

Not yet arena-confirmed at all, individually or combined - deliberately, per
this session's own explicit choice to skip further arena time given a real
tournament deadline. Correctness is verified to a very high standard for
each piece (bit-exact where possible, debug-assert against ground truth
where not); playing-strength effect is not measured and is honestly
unknown. This is the version in submission.zip as of this session.
