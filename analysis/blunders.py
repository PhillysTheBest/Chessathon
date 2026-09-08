"""Replay a rated game and find the moves that actually lost it.

Two-stage design, so a full game analyses in minutes instead of hours:

  Stage 1 (cheap, every one of our moves): one FIXED-DEPTH root search over
  every legal move (_shallow_root_scores below), sharing a single TT/killers/
  history pass across the whole position the way search_root's own root loop
  (_root_window in agent.py) does in real play. This is vastly cheaper than
  scoring every legal move independently, because real alpha-beta pruning
  discards most of the tree - that is the entire point of the search already
  built. It gives an exact score for its own top pick and, for the rest, a
  PVS null-window result: exact if it beat alpha, otherwise only an upper
  bound. Fine for ranking; never fine for a verdict.

  Stage 2 (expensive, only positions Stage 1 flags as suspicious): the
  original, proven-sound comparison - push a candidate, search the child to
  a fixed depth, negate - applied only to a handful of candidates (the
  played move, Stage 1's own pick, and the next few by Stage 1's score), not
  every legal move. TT/history/killers/counters are reset immediately before
  EACH candidate here, not once per position like the old version of this
  script - so no candidate is advantaged by ordering information left behind
  by scoring the one before it.

A verdict is ALWAYS a same-depth child-search comparison, nothing else.
Comparing a parent search's score to a child search's score is never done
anywhere in this file: that produced 300cp of phantom loss on moves that
were actually best, in earlier testing of this same script.

A hard limit of this design, stated plainly: Stage 1's shallow depth cannot
see a problem that only reveals itself deeper - the round 46 / round 62
failure mode this project has already documented in analysis/positions.epd
(a move that looks fine through depth 7-8 and loses to a forced sequence
only visible at depth 9+). A position with no Stage-1 disagreement and no
eval-swing trigger is *not* run through Stage 2 at all, so a genuine horizon
blunder can pass through this analyser completely undetected. Use
--force-deep to run Stage 2 on every move in a range regardless of what
Stage 1 says, for when a horizon blunder is suspected on other grounds (a
rated loss with no other explanation, say).

    uv run python -m analysis.blunders game.pgn --us "Istanbul's finest"
    uv run python -m analysis.blunders game.pgn --us "..." --from-move 13 --to-move 22 --force-deep
"""

import argparse
import io as _io
import sys
import time
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent

BUDGET_S = 600.0


def _reset_search_state():
    """Wipe every piece of search state this script's searches read from, so
    two calls never share ordering information they shouldn't. Real play
    never does this between moves (TT/history/killers/counters are
    deliberately persistent module-level globals) - this script does it more
    aggressively than real play on purpose, for reproducibility and fairness
    between candidates, at the cost of not reflecting how a real game's
    accumulated history would have ordered these exact searches.
    """
    agent.tt_clear()
    agent.SEARCH.game_keys = {}
    agent.SEARCH.path = []
    agent.SEARCH.nodes = 0
    agent.SEARCH.hard_deadline = time.perf_counter() + BUDGET_S
    agent.SEARCH.soft_deadline = agent.SEARCH.hard_deadline
    for row in agent.KILLERS:
        row[0] = 0
        row[1] = 0
    for pt in range(7):
        for f in range(64):
            for t in range(64):
                agent.HISTORY[pt][f][t] = 0
    for f in range(64):
        for t in range(64):
            agent.COUNTER[f][t] = 0


def _shallow_root_scores(board, depth):
    """Stage 1. One fixed-depth pass over every legal move sharing one
    TT/killers/history state, mirroring _root_window's own PVS algorithm in
    agent.py without calling or modifying that function - this script only
    needs the extra visibility into every move's score, which _root_window
    does not expose (by design: search_root only ever needs the winner).

    Returns (best_move, best_score, scores) where scores maps every legal
    move to what this pass found for it. best_move/best_score are exact.
    Every other entry is a PVS result: exact only if it beat the running
    alpha at the time it was searched, otherwise a fail-low upper bound -
    good enough to rank candidates, not to compare with the specificity a
    verdict requires (see Stage 2 below).
    """
    _reset_search_state()
    moves = list(board.legal_moves)
    ordered = agent.order_moves(board, moves, None, 0, None)
    base_stack = len(board.move_stack)

    alpha, beta = -agent.INFINITY, agent.INFINITY
    scores: dict[chess.Move, int] = {}
    best_move, best_score = ordered[0], -agent.INFINITY
    for index, move in enumerate(ordered):
        board.push(move)
        try:
            if index == 0:
                score = -agent.negamax(board, depth - 1, -beta, -alpha, 1, move)
            else:
                score = -agent.negamax(board, depth - 1, -alpha - 1, -alpha, 1, move)
                if alpha < score < beta:
                    score = -agent.negamax(board, depth - 1, -beta, -alpha, 1, move)
        except agent.TimeUp:
            continue  # scores[move] just never gets set for this one
        finally:
            # TimeUp deep in negamax leaks every push in flight, not just
            # this move's (negamax's own move loops push before recursing
            # and pop after, with no finally of their own) - unwind to
            # base_stack, not one pop, or every later position in this game
            # gets analysed against a corrupted board. Mirrors
            # _root_window's own finally in agent.py exactly.
            while len(board.move_stack) > base_stack:
                board.pop()
        scores[move] = score
        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
    return best_move, best_score, scores


def _verify_candidates(board, candidates, depth, *, verbose=False):
    """Stage 2. The original equal-depth comparison - push, search the child
    to `depth`, negate - for each of `candidates` only. State is reset
    before every single candidate, not once for the whole batch.

    Check extensions and the forced-reply extension can chain in a heavily
    forcing position (a long check-and-only-one-legal-reply sequence keeps
    extending the search well past its nominal depth), so each candidate is
    timed and, with verbose on, printed as it finishes - the only way to
    tell "still working through a sharp position" from "stuck" without that.
    BUDGET_S (600s) is a safety ceiling, not a target; a candidate that hits
    it is reported as unverified rather than crashing the run - the played
    move being unverifiable means this position can't produce a verdict at
    all (see main()), but one alternative candidate timing out just drops
    that one candidate from consideration, since the others still verified
    cleanly.
    """
    results = {}
    base_stack = len(board.move_stack)
    for move in candidates:
        c0 = time.perf_counter()
        _reset_search_state()
        board.push(move)
        try:
            score = -agent.negamax(
                board, depth - 1, -agent.INFINITY, agent.INFINITY, 0, None)
        except agent.TimeUp:
            elapsed = time.perf_counter() - c0
            print(f"      ...{board.san(move):8s} UNVERIFIED - hit the "
                  f"{BUDGET_S:.0f}s safety ceiling ({elapsed:.1f}s), likely a "
                  f"forced-reply/check-extension chain", flush=True)
            continue
        finally:
            # See the class of bug this guards against in
            # _shallow_root_scores' matching comment: unwind to base_stack,
            # never a single pop, after a TimeUp.
            while len(board.move_stack) > base_stack:
                board.pop()
        results[move] = score
        if verbose:
            print(f"      ...{board.san(move):8s} {score:+6d}  "
                  f"({time.perf_counter() - c0:.1f}s)", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Find the losing moves in a game.")
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--us", required=True)
    parser.add_argument("--shallow-depth", type=int, default=4,
                         help="Stage 1 fixed depth, applied to every one of our moves.")
    parser.add_argument("--deep-depth", type=int, default=7,
                         help="Stage 2 fixed depth, applied only to suspicious positions.")
    parser.add_argument("--candidates", type=int, default=4,
                         help="how many of Stage 1's top-scoring moves (plus the "
                              "played move and Stage 1's own pick) get Stage 2 "
                              "verification at a suspicious position.")
    parser.add_argument("--threshold", type=int, default=150)
    parser.add_argument("--swing-threshold", type=int, default=120,
                         help="flag a position as suspicious if the position's "
                              "value for us dropped by at least this much over "
                              "our move and the opponent's reply, even when "
                              "Stage 1's pick matched what was played. "
                              "HEURISTIC, not a proof of anything.")
    parser.add_argument("--from-move", type=int, default=0)
    parser.add_argument("--to-move", type=int, default=10**9)
    parser.add_argument("--force-deep", action="store_true",
                         help="run Stage 2 on every move in range regardless of "
                              "what Stage 1 says - for when a horizon blunder is "
                              "suspected on other grounds and a shallow scan "
                              "might miss it entirely.")
    parser.add_argument("--quiet", action="store_true",
                         help="suppress per-candidate Stage 2 progress lines - "
                              "off by default because a sharp/forcing position "
                              "can make one candidate's fixed-depth search take "
                              "much longer than usual (check extensions keep "
                              "extending it), and with no progress output that "
                              "looks identical to the process being stuck.")
    args = parser.parse_args()

    game = chess.pgn.read_game(_io.StringIO(args.pgn.read_text()))
    if game is None:
        raise SystemExit(f"no game in {args.pgn}")

    our_colour = (chess.WHITE if args.us.lower() in game.headers.get("White", "").lower()
                  else chess.BLACK)
    print(f"{game.headers.get('White')} vs {game.headers.get('Black')} "
          f"({game.headers.get('Result')})")
    print(f"we are {'White' if our_colour else 'Black'}, "
          f"stage1 depth {args.shallow_depth}, stage2 depth {args.deep_depth}\n")

    board = game.board()
    findings = []
    prev_best_score = None  # Stage 1's best_score two of our plies ago
    for move in game.mainline_moves():
        in_range = args.from_move <= board.fullmove_number <= args.to_move
        if board.turn != our_colour or not in_range:
            board.push(move)
            continue

        print(f"  checking move {board.fullmove_number}...", flush=True)
        t0 = time.perf_counter()
        best_move, best_score, all_scores = _shallow_root_scores(board, args.shallow_depth)
        played_shallow = all_scores.get(move)

        swing = (prev_best_score - best_score) if prev_best_score is not None else None
        disagreement = best_move != move
        suspicious = args.force_deep or disagreement or (
            swing is not None and swing >= args.swing_threshold)

        if suspicious:
            ranked = sorted(all_scores.items(), key=lambda kv: -kv[1])
            candidates = {move, best_move}
            for m, _s in ranked[:args.candidates]:
                candidates.add(m)
            verified = _verify_candidates(board, candidates, args.deep_depth,
                                           verbose=not args.quiet)
            if move not in verified:
                # The played move itself hit the safety ceiling - nothing
                # sound to compare it against, so this position produces no
                # verdict rather than a wrong one.
                print(f"??? {board.fullmove_number:3d}. {board.san(move):8s} "
                      f"UNVERIFIABLE - played move itself hit the safety "
                      f"ceiling", flush=True)
                prev_best_score = None
                board.push(move)
                continue
            deep_best_move = max(verified, key=lambda m: verified[m])
            deep_best_score = verified[deep_best_move]
            played_score = verified[move]
            loss = deep_best_score - played_score
            elapsed = time.perf_counter() - t0
            flag = "!!" if loss >= args.threshold else "??"
            reason = "disagreement" if disagreement else (
                f"swing {swing}" if swing is not None and swing >= args.swing_threshold
                else "forced")
            print(f"{flag} {board.fullmove_number:3d}. {board.san(move):8s} "
                  f"{played_score:+6d}   best {board.san(deep_best_move):8s} "
                  f"{deep_best_score:+6d}   loss {loss:5d}   "
                  f"[{reason}, {len(candidates)} verified, {elapsed:.1f}s]",
                  flush=True)
            if loss >= args.threshold:
                findings.append((board.fullmove_number, board.san(move),
                                  board.san(deep_best_move), loss, played_score,
                                  deep_best_score))
            prev_best_score = -deep_best_score
        else:
            elapsed = time.perf_counter() - t0
            print(f"   {board.fullmove_number:3d}. {board.san(move):8s} "
                  f"{played_shallow:+6d}   (matches stage-1 best, no verify, "
                  f"{elapsed:.1f}s)", flush=True)
            prev_best_score = -played_shallow

        board.push(move)

    print("=" * 62)
    if not findings:
        print(f"nothing verified at {args.threshold}cp or more loss")
        return
    print(f"{len(findings)} move(s) lost {args.threshold}cp or more, worst first:\n")
    for num, played, best, loss, ps, bs in sorted(findings, key=lambda f: -f[3])[:10]:
        print(f"  move {num:3d}  played {played:9s} ({ps:+6d})   "
              f"best {best:9s} ({bs:+6d})   -{loss}cp")


if __name__ == "__main__":
    main()
