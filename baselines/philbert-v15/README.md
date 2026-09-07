Frozen v1.5: v1.4 plus aspiration windows and the ebf figure in debug output.

Measured so far, by node count to a fixed depth 8 over four positions:
  289804 -> 244629 nodes, -15.6%

Its 40 game arena against philbert-v14 was still running when this was frozen,
so the strength number is not in yet. Do not ship it on the node count alone;
SEE also halved the tree once and won nothing until its threshold was fixed.
