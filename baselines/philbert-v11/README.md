Frozen copy of agent.py at commit 0776a5f, kept as a regression opponent.
This is the current ladder version: v2 has to beat it before it ships.

    uv run python -m harness.arena --opponent baselines/philbert-v11 --games 40

Measured at 10s+0.1s when it was frozen:
    vs baselines/philbert-v1   +20 =4 -6 over 30 games, 73.3%
    vs baselines/minimax       +10 =0 -0 over 10 games, 100%

Do not edit. To roll the baseline forward, re-snapshot from a later commit.
