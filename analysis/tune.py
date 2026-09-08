"""Build single-constant variants of agent.py for A/B testing.

The SEE threshold was worth 26 percentage points and every other pruning
constant in the engine was guessed once, in one pass, and never questioned.
This makes each one an experiment instead of an assumption.

    uv run python -m analysis.tune --list
    uv run python -m analysis.tune rfp-40 rfp-120

Each variant is a full copy of agent.py with one line rewritten, dropped in
baselines/tune-<name>/ with DEBUG off, ready for harness.arena.
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name -> (regex matching the whole line, replacement line)
VARIANTS: dict[str, tuple[str, str]] = {
    # Reverse futility: how far above beta we must be to cut without searching.
    "rfp-40":      (r"^RFP_MARGIN = .*",     "RFP_MARGIN = 40"),
    "rfp-120":     (r"^RFP_MARGIN = .*",     "RFP_MARGIN = 120"),

    # Late move pruning: how many quiet moves to try before giving up on a node.
    "lmp-tight":   (r"^LMP_COUNT = .*",      "LMP_COUNT = [0, 3, 5, 8, 13, 19, 26]"),
    "lmp-loose":   (r"^LMP_COUNT = .*",      "LMP_COUNT = [0, 8, 12, 18, 27, 38, 52]"),

    # Futility: quiet moves cannot rescue a node this far below alpha.
    "fut-tight":   (r"^FUTILITY_MARGIN = .*",
                    "FUTILITY_MARGIN = [0, 70, 130, 200, 290, 390, 500]"),
    "fut-loose":   (r"^FUTILITY_MARGIN = .*",
                    "FUTILITY_MARGIN = [0, 150, 260, 400, 570, 770, 1000]"),

    # Delta pruning in quiescence.
    "delta-100":   (r"^DELTA_MARGIN = .*",   "DELTA_MARGIN = 100"),
    "delta-350":   (r"^DELTA_MARGIN = .*",   "DELTA_MARGIN = 350"),

    # Late move reductions. The divisor sets how aggressively late quiet moves
    # are searched shallower; smaller divisor means deeper reductions.
    "lmr-aggr":    (r"^        LMR\[_d\]\[_i\] = .*",
                    "        LMR[_d][_i] = int(0.75 + math.log(_d) * math.log(_i) / 1.90)"),
    "lmr-shy":     (r"^        LMR\[_d\]\[_i\] = .*",
                    "        LMR[_d][_i] = int(0.75 + math.log(_d) * math.log(_i) / 2.80)"),

    # SEE cutoff either side of the -110 that won, to check it is a real optimum
    # and not one lucky sample.
    "see-60":      (r"^QS_SEE_THRESHOLD = .*", "QS_SEE_THRESHOLD = -60"),
    "see-200":     (r"^QS_SEE_THRESHOLD = .*", "QS_SEE_THRESHOLD = -200"),

    # Bounded gravity history update + malus, replacing the flat unbounded
    # `+= depth * depth` accumulate. See HISTORY_AGING_ENABLED's own comment.
    "history-aging": (r"^HISTORY_AGING_ENABLED = .*",
                       "HISTORY_AGING_ENABLED = True"),
}


def build(name: str) -> Path:
    pattern, replacement = VARIANTS[name]
    source = (ROOT / "agent.py").read_text(encoding="utf-8")
    patched, count = re.subn(pattern, replacement, source, count=1, flags=re.M)
    if count != 1:
        raise SystemExit(f"{name}: pattern {pattern!r} matched {count} lines, expected 1")
    patched = patched.replace("DEBUG = True", "DEBUG = False", 1)

    out = ROOT / "baselines" / f"tune-{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent.py").write_text(patched, encoding="utf-8", newline="\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tuning variants.")
    parser.add_argument("names", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, (_pattern, replacement) in VARIANTS.items():
            print(f"  {name:12s} {replacement}")
        return

    names = list(VARIANTS) if args.all else args.names
    if not names:
        parser.error("give variant names, or --all, or --list")
    for name in names:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}; try --list")
        print(f"  built {build(name)}")


if __name__ == "__main__":
    main()

