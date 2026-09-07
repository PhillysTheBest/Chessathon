"""Replay a rated game and find the moves that actually lost it.

At every position where it was our turn, search to a fixed depth to get the
best move, then score BOTH the move we played and that best move the same way:
push it, search the child to the same depth, negate. Comparing two child
searches at equal depth is apples to apples. Comparing a child search against a
parent search is not, and shows 300cp of phantom loss on moves that were
actually best.

Fixed depth rather than fixed time, so a rerun gives the same answer.

    uv run python -m analysis.blunders game.pgn --us "Istanbul's finest" --depth 7
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


def search_to_depth(board, depth):
    """Score the position at a fixed depth, from the side to move's view."""
    agent.SEARCH.hard_deadline = time.perf_counter() + BUDGET_S
    agent.SEARCH.game_keys = {}
    agent.SEARCH.path = []
    agent.SEARCH.nodes = 0
    return agent.negamax(board, depth, -agent.INFINITY, agent.INFINITY, 0, None)


def score_child(board, move, depth):
    """Score `move` by searching the position it leads to, at `depth`."""
    board.push(move)
    try:
        return -search_to_depth(board, depth)
    finally:
        board.pop()


def main():
    parser = argparse.ArgumentParser(description="Find the losing moves in a game.")
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--us", required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--threshold", type=int, default=150)
    parser.add_argument("--from-move", type=int, default=0)
    args = parser.parse_args()

    game = chess.pgn.read_game(_io.StringIO(args.pgn.read_text()))
    if game is None:
        raise SystemExit(f"no game in {args.pgn}")

    our_colour = (chess.WHITE if args.us.lower() in game.headers.get("White", "").lower()
                  else chess.BLACK)
    print(f"{game.headers.get('White')} vs {game.headers.get('Black')} "
          f"({game.headers.get('Result')})")
    print(f"we are {'White' if our_colour else 'Black'}, fixed depth {args.depth}\n")

    board = game.board()
    findings = []
    for move in game.mainline_moves():
        if board.turn != our_colour or board.fullmove_number < args.from_move:
            board.push(move)
            continue

        agent.tt_clear()
        best, _, _, _ = agent.search_root(board.copy(stack=False), BUDGET_S, BUDGET_S) \
            if False else (None, 0, 0, "")

        # Best move at this depth, found by scoring every legal move the same way.
        agent.tt_clear()
        scored = [(score_child(board, m, args.depth - 1), m) for m in board.legal_moves]
        best_score, best = max(scored, key=lambda p: p[0])
        played_score = next(s for s, m in scored if m == move)

        loss = best_score - played_score
        flag = "!!" if loss >= args.threshold else "  "
        if loss >= args.threshold:
            findings.append((board.fullmove_number, board.san(move), board.san(best),
                             loss, played_score, best_score))
        print(f"{flag} {board.fullmove_number:3d}. {board.san(move):8s} {played_score:+6d}"
              f"   best {board.san(best):8s} {best_score:+6d}   loss {loss:5d}")
        board.push(move)

    print("=" * 62)
    if not findings:
        print(f"nothing lost {args.threshold}cp or more at depth {args.depth}")
        return
    print(f"{len(findings)} move(s) lost {args.threshold}cp or more, worst first:\n")
    for num, played, best, loss, ps, bs in sorted(findings, key=lambda f: -f[3])[:10]:
        print(f"  move {num:3d}  played {played:9s} ({ps:+6d})   "
              f"best {best:9s} ({bs:+6d})   -{loss}cp")


if __name__ == "__main__":
    main()
