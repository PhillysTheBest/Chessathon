"""Run the scenario suite and report what the engine gets right.

A test suite is how you turn "it played something odd" into something you can
measure. Each line is a position plus either a move the engine must find (bm)
or one it must not play (am). Every rated game that shows a weakness should add
a line, so the same mistake cannot come back unnoticed.

    uv run python -m analysis.suite --seconds 3
    uv run python -m analysis.suite --theme kingsafety --seconds 5
"""

import argparse
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent

SUITE = Path(__file__).resolve().parent / "positions.epd"

agent.DEBUG = False


def normalise(san):
    """Ra8+ and Ra8# are the same move; the suffix is annotation, not identity."""
    return san.rstrip("+#")


def parse(line):
    parts = [p.strip() for p in line.split(";")]
    entry = {"fen": parts[0]}
    for field in parts[1:]:
        if field.startswith("id "):
            entry["id"] = field[3:].strip().strip('"')
        elif field.startswith("theme "):
            entry["theme"] = field[6:].strip()
        elif field.startswith("bm "):
            entry["bm"] = field[3:].strip()
        elif field.startswith("am "):
            entry["am"] = field[3:].strip()
    return entry


def main():
    parser = argparse.ArgumentParser(description="Run the scenario suite.")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--theme")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = [parse(ln) for ln in SUITE.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    if args.theme:
        cases = [c for c in cases if c.get("theme") == args.theme]

    by_theme: dict[str, list[bool]] = {}
    failures = []

    for case in cases:
        board = chess.Board(case["fen"])
        agent.tt_clear()
        agent.GAME_KEYS.clear()
        agent.SEARCH.game_keys = agent.GAME_KEYS
        started = time.perf_counter()
        move, score, depth, _ = agent.search_root(
            board.copy(stack=False), args.seconds * 0.6, args.seconds)
        elapsed = time.perf_counter() - started
        san = board.san(move)

        if "bm" in case:
            ok = normalise(san) == normalise(case["bm"])
            want = f"play {case['bm']}"
        else:
            ok = normalise(san) != normalise(case["am"])
            want = f"avoid {case['am']}"

        by_theme.setdefault(case.get("theme", "other"), []).append(ok)
        if not ok:
            failures.append((case["id"], want, san, score, depth))
        if args.verbose or not ok:
            print(f"  {'PASS' if ok else 'FAIL'}  {case['id']:34s} "
                  f"{want:14s} played {san:8s} ({score:+5d}, d{depth}, {elapsed:.1f}s)")

    print()
    total = passed = 0
    for theme, results in sorted(by_theme.items()):
        p, n = sum(results), len(results)
        total += n
        passed += p
        print(f"  {theme:14s} {p}/{n}")
    print(f"  {'TOTAL':14s} {passed}/{total}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
