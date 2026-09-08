import argparse
import math
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100

# SPRT defaults, sized for THIS project's sample sizes, not fishtest's. Big
# engine-testing projects run elo0=0/elo1=5 because they can afford tens of
# thousands of games; at our own noise floor (~7-8% SE at n=40, roughly +/-55
# elo of resolving power per CLAUDE.md), a 5-elo alternative would need
# several hundred games to ever cross a boundary. elo1=40 is sized to resolve
# an effect on the order of what actually shipped or got rejected in this
# repo's history (SEE's swing, RFP's swing) within a two-to-one multiple of
# the existing --games budget, not a fixed academic convention.
SPRT_ELO0 = 0.0
SPRT_ELO1 = 40.0
SPRT_ALPHA = 0.05
SPRT_BETA = 0.05
SPRT_MIN_GAMES = 8


def elo_to_score(elo: float) -> float:
    """Expected score for an `elo`-point advantage, the standard logistic model."""
    return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))


def sprt_llr(total_score: float, n: int, p0: float, p1: float) -> float:
    """Wald SPRT log-likelihood ratio, approximating each game's score (0, 0.5 or 1)
    as i.i.d. Normal(mu, sigma^2) with sigma^2 fixed at a Bernoulli(p_mid) trial's
    variance. Draws reduce the true variance below this, so the approximation is
    conservative (stops later than a full trinomial model would, never earlier) -
    a reasonable trade for staying simple enough to verify by hand.
    """
    p_mid = (p0 + p1) / 2.0
    variance = max(p_mid * (1.0 - p_mid), 1e-6)
    return (p1 - p0) / variance * (total_score - n * p_mid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an agent over several games.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument(
        "--sprt", action="store_true",
        help="Stop before --games once the result is decisively one-sided "
             "(Wald SPRT approximation), instead of always playing every game. "
             "--games is still the hard cap. Off by default: every existing "
             "invocation keeps its current fixed-N behavior.",
    )
    parser.add_argument("--elo0", type=float, default=SPRT_ELO0, help="SPRT null hypothesis.")
    parser.add_argument("--elo1", type=float, default=SPRT_ELO1, help="SPRT alternative.")
    parser.add_argument("--alpha", type=float, default=SPRT_ALPHA, help="SPRT false-positive rate.")
    parser.add_argument("--beta", type=float, default=SPRT_BETA, help="SPRT false-negative rate.")
    parser.add_argument(
        "--sprt-min-games", type=int, default=SPRT_MIN_GAMES,
        help="Games to play before SPRT may stop early, so an unlucky start can't decide it.",
    )
    arguments = parser.parse_args()

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    total_score = 0.0

    p0 = elo_to_score(arguments.elo0)
    p1 = elo_to_score(arguments.elo1)
    upper = math.log((1 - arguments.beta) / arguments.alpha)
    lower = math.log(arguments.beta / (1 - arguments.alpha))
    stopped_early = False

    for game in range(arguments.games):
        plays_white = game % 2 == 0
        white, black = (agent, opponent) if plays_white else (opponent, agent)
        outcome = play_match(
            local(white),
            local(black),
            arguments.base_ms,
            arguments.increment_ms,
            ply_cap=arguments.ply_cap,
        )
        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        if outcome.result == "draw" or outcome.result == "void":
            draws += 1
            total_score += 0.5
        elif (outcome.result == "white") == plays_white:
            wins += 1
            total_score += 1.0
        else:
            losses += 1
        n = game + 1
        print(f"game {n}/{arguments.games}: {outcome.result} by {outcome.termination}")

        if arguments.sprt and n >= arguments.sprt_min_games:
            llr = sprt_llr(total_score, n, p0, p1)
            if llr >= upper:
                print(f"\nSPRT: accept H1 (elo >= {arguments.elo1}) after {n} games, llr={llr:.2f}")
                stopped_early = True
                break
            if llr <= lower:
                print(f"\nSPRT: accept H0 (elo <= {arguments.elo0}) after {n} games, llr={llr:.2f}")
                stopped_early = True
                break

    played = wins + draws + losses
    score = (wins + draws / 2) / played
    label = f"{played} games" + (" (stopped early)" if stopped_early else "")
    print(f"\n{arguments.agent} vs {arguments.opponent} over {label}")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print("terminations: " + ", ".join(f"{name} {count}" for name, count in terminations.items()))
    broken = {name: count for name, count in terminations.items() if name in FAILED_TERMINATIONS}
    if broken:
        raise SystemExit(
            "your agent failed to finish a game: "
            + ", ".join(f"{name} {count}" for name, count in broken.items())
        )


if __name__ == "__main__":
    main()
