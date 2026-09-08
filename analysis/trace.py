"""Replay a rated game up to a specific one of our moves, using the real
`get_move()` machinery (so TT/killers/history/counters/OpponentTracker end
up in the same state real play would have left them in — the current
analysis tools deliberately reset all of that per position for fair
candidate comparison, which is the wrong choice for this tool), and the real
clock the match log/PGN recorded before that move. Then search that exact
position with search_root's trace hook, showing every root candidate's
score/nodes/time/completion/telemetry at every depth — not just the winner.

This is the tool for the question analysis.blunders cannot answer: did the
real search evaluate the objectively-better move above the one it played but
still return the worse one (a root/time-management bug), or did it genuinely
settle on the worse move as best (an ordering/evaluation problem)? Those are
different problems with different fixes, and until this trace exists there
is no way to tell them apart for a real tournament game - the platform's
stderr log truncates exactly the part of a real game that would show it.

Fidelity caveat, stated plainly: replaying via real get_move() calls uses
real wall-clock timing, which depends on this machine's speed and current
load - not necessarily identical to what the original game's search reached
at each earlier move. If a replayed move doesn't match what the PGN
actually recorded, that's reported as a warning (not a crash): the position
still advances via the PGN's real move, but the TT/history state feeding
into later moves may have diverged slightly from what real play accumulated.
Close enough for this tool's purpose in every position tried so far; not a
guarantee.

    uv run python -m analysis.trace game.pgn --us "Istanbul's finest" --move 17
    uv run python -m analysis.trace game.pgn --us "..." --move 17 --candidates Bxe4,Bg4
"""

import argparse
import io as _io
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent

STARTING_CLOCK_MS = 120_000.0  # matches CLAUDE.md's documented base clock


def replay_to(game, our_colour, target_move_number):
    """Push every move up to (not including) our move `target_move_number`,
    calling the real agent.get_move() for each of our own earlier moves so
    TT/killers/history/counters/OpponentTracker end up in a real-play state.
    Returns (board, our_time_left_ms, mismatches) - the position, the real
    clock entering the target move, and how many earlier replayed moves
    didn't match what the PGN recorded (see this module's fidelity caveat).
    """
    board = game.board()
    our_time_left_ms = STARTING_CLOCK_MS
    node = game
    mismatches = 0

    for move in game.mainline_moves():
        if board.fullmove_number == target_move_number and board.turn == our_colour:
            return board, our_time_left_ms, mismatches

        if board.turn == our_colour:
            # get_move() runs the real search (updating TT/killers/history/
            # counters and TRACKER as a side effect, exactly like real play)
            # and returns its own choice of move - which should be `move`,
            # the PGN's actual recorded move, but isn't guaranteed to be (see
            # this module's fidelity caveat). Advance the board with the
            # PGN's real move regardless, so the replay always ends up at
            # the real game's real positions even if a mismatch happened.
            predicted = agent.get_move(board.fen(), int(our_time_left_ms))
            if predicted != move.uci():
                mismatches += 1
                print(f"  [replay warning] move {board.fullmove_number}: "
                      f"real game played {board.san(move)}, replay's own "
                      f"search preferred {predicted} - continuing with the "
                      f"PGN's actual move (see this file's fidelity caveat)",
                      file=sys.stderr)
        board.push(move)

        node = node.variations[0]
        clk = node.clock()
        moved_colour = not board.turn  # side that just moved (board.turn
                                        # already flipped by the push above)
        if clk is not None and moved_colour == our_colour:
            our_time_left_ms = clk * 1000.0

    raise SystemExit(
        f"move {target_move_number} for {'White' if our_colour else 'Black'} "
        f"never reached in this game")


def format_info(move_san, info, is_watched):
    tel = info["telemetry"]
    mark = "*" if is_watched else " "
    tt = "TT" if info["is_first"] else "  "
    status = "ok" if info["completed"] else "TIMEOUT"
    score = "n/a" if info["score"] is None else f"{info['score']:+5d}"
    return (f"  {mark}[{tt}] {move_san:8s} score {score}  nodes {info['nodes']:6d}  "
            f"time {info['elapsed_s']*1000:6.1f}ms  {status:7s}  "
            f"cutoffs(idx0-3+)={tel['cutoff_at_index']} nmp={tel['nmp_cutoffs']}/"
            f"{tel['nmp_tries']} lmr_re={tel['lmr_full_researches']}/"
            f"{tel['lmr_reductions']}")


def main():
    parser = argparse.ArgumentParser(
        description="Trace a real search decision, depth by depth, under "
                     "real replayed search state and the real clock.")
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--us", required=True)
    parser.add_argument("--move", type=int, required=True,
                         help="our fullmove number to trace")
    parser.add_argument("--candidates", default="",
                         help="comma-separated SAN moves to always show, "
                              "e.g. Bxe4,Bg4. The played move is always shown.")
    parser.add_argument("--all", action="store_true",
                         help="show every root candidate, not just the "
                              "played move and --candidates.")
    args = parser.parse_args()

    game = chess.pgn.read_game(_io.StringIO(args.pgn.read_text()))
    if game is None:
        raise SystemExit(f"no game in {args.pgn}")
    our_colour = (chess.WHITE if args.us.lower() in game.headers.get("White", "").lower()
                  else chess.BLACK)

    # Find the actual move played at args.move, for reporting and for the
    # "always show the played move" rule.
    played_move = None
    scratch = game.board()
    for move in game.mainline_moves():
        if scratch.fullmove_number == args.move and scratch.turn == our_colour:
            played_move = move
            break
        scratch.push(move)
    if played_move is None:
        raise SystemExit(f"move {args.move} not found for {args.us}")
    played_san = scratch.san(played_move)

    print(f"Replaying to move {args.move} via real get_move() calls "
          f"(real TT/killers/history/counters, real clock)...", file=sys.stderr)
    board, time_left_ms, mismatches = replay_to(game, our_colour, args.move)
    print(f"Reached move {args.move}, {'White' if board.turn else 'Black'} to "
          f"move, clock {time_left_ms:.0f}ms, {mismatches} replay mismatch(es) "
          f"along the way\nFEN: {board.fen()}\n")

    watched_san = {played_san} | {c.strip() for c in args.candidates.split(",") if c.strip()}
    watched_uci = set()
    for san in watched_san:
        try:
            watched_uci.add(board.parse_san(san).uci())
        except ValueError:
            print(f"  [warning] --candidates entry {san!r} is not a legal "
                  f"move here, ignoring", file=sys.stderr)

    soft_s, hard_s = agent.compute_budget(board, time_left_ms)
    print(f"Budget: soft {soft_s*1000:.0f}ms hard {hard_s*1000:.0f}ms "
          f"(matches what get_move would have used for this exact clock)\n")

    events_by_depth: dict[int, list] = {}

    def trace(depth, index, move, info):
        events_by_depth.setdefault(depth, []).append((index, move, info))

    move, score, depth, pv = agent.search_root(board, soft_s, hard_s, trace=trace)

    for d in sorted(events_by_depth):
        print(f"depth {d}:")
        for _index, cand_move, info in events_by_depth[d]:
            uci = cand_move.uci()
            if not args.all and uci not in watched_uci:
                continue
            san = board.san(cand_move)
            print(format_info(san, info, uci in watched_uci))
        print()

    print(f"Real search's final choice: {board.san(move)} depth {depth} "
          f"score {score:+d}")
    print(f"PGN's actual move: {played_san}")
    if move == played_move:
        print("MATCH - the real search's own choice is what was played.")
    else:
        print("MISMATCH - the real search preferred a different move than "
              "what the PGN recorded. If this trace's own final choice "
              "differs from the played move, the replay's search state or "
              "timing has diverged from the original game (see the "
              "fidelity caveat in this file's docstring) - treat the "
              "per-depth numbers above as informative, not as literally "
              "what happened in the original game.")
    print(f"PV from here: {pv}")


if __name__ == "__main__":
    main()
