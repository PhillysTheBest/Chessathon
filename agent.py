"""PhilbertV1 - AI Chessathon submission.

An 8x8 adaptation of the 5x5 variant engine kept in Philbert/PhilbertV1.py.

The SEARCH is carried over from that engine more or less intact: principal
variation search over negamax, a transposition table with exact/lower/upper
flags and ply-adjusted mate scores, internal iterative deepening, quiescence at
the leaves, null move pruning, late move reductions, late move pruning,
futility pruning, check extensions, and MVV-LVA / killer / counter move /
history ordering, all driven by iterative deepening under a soft/hard time
split that unwinds through a TimeUp exception.

The BOARD is not carried over. The 5x5 engine hand-rolled its own bitboards for
a 25 square board with a rook+knight compound piece, no castling, no en passant
and no double pawn push. None of that generalises, and an illegal move is an
instant loss, so movegen and legality come from python-chess instead.

Three things from the 5x5 version are deliberately NOT reproduced, because they
were right there and are wrong here:

  * the rook was worth 850 (it was a chancellor, nearly a queen). Here it is a
    rook, worth about 500.
  * stalemate was a LOSS in that variant. Here it is a draw, scored 0.
  * mate scores were folded by ply // 2. Here they use ply directly.

Everything at module scope runs inside the 90 second init budget, before the
clock starts.
"""

import contextlib
import io
import math
import pkgutil
import sys
import time

import chess
import chess.polyglot
import numpy as np
from numba import njit

# =============================================================================
# TUNABLES
# =============================================================================

DEBUG = True  # writes to stderr; the platform keeps it in the per-game log

# Stamped with the current git commit by harness.package's build step (see
# that file) so a rated game's log always says exactly which commit played
# it - resolved a real ambiguity this session about which build produced a
# given match log. "dev" means this copy was never rebuilt through that path
# (e.g. run directly from the working tree during development).
BUILD_ID = "dev"

MAX_DEPTH_CAP = 64
MAX_PLY = 128

MATE_SCORE = 30000
MATE_BOUND = MATE_SCORE - 1000  # scores past this are forced mates

INFINITY = 1 << 30

# Clock. The referee measures wall time and the watchdog does not forgive, so
# every budget below is clamped against SAFETY_MARGIN_MS before it is used.
SAFETY_MARGIN_MS = 250.0
INCREMENT_MS = 500.0
# Below this the clock decides the budget, not the position. A third of a two
# second clock per move is how a playable game becomes a flag, and a flag is a
# loss. Under the increment the clock actually recovers.
LOW_CLOCK_MS = 20000.0

# Aspiration windows start at depth 5, where the prior iteration is useful as
# a score estimate. A failed window is never accepted as an iteration result:
# the loop widens and re-searches, and force-checks the hard deadline before
# every attempt so a string of re-searches cannot flag the game.
ASPIRATION_DELTA = 25
ASPIRATION_MIN_DEPTH = 5

# A compact project-trained neural evaluator can be packaged at this path. It
# is deliberately optional: without a validated, locally trained weight file,
# the hand-written evaluator remains the source of truth.
NNUE_RESOURCE = "weights/nnue.npz"

# 32M entries * 15 bytes = about 480 MiB, leaving ample room within the 2 GiB
# competition limit while preventing a single deep search from thrashing the TT.
TT_SIZE = 1 << 25
TT_MASK = TT_SIZE - 1
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# =============================================================================
# PIECE VALUES (tapered)
# =============================================================================
# Indexed by chess.PAWN..chess.KING (1..6); slot 0 is unused so a piece type
# indexes directly, the same trick the 5x5 engine used with its own indices.

PIECE_VALUE_MG = [0, 82, 337, 365, 477, 1025, 0]
PIECE_VALUE_EG = [0, 94, 281, 297, 512, 936, 0]

# Phase weights: knight/bishop 1, rook 2, queen 4. Two of each per side = 24.
PHASE_WEIGHT = [0, 0, 1, 1, 2, 4, 0]
MAX_PHASE = 24

# =============================================================================
# PIECE SQUARE TABLES
# =============================================================================
# Written rank 8 first so they read like a board. _build flips them into
# python-chess square indices (0 = a1, 63 = h8).

PAWN_MG = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [98, 134, 61, 95, 68, 126, 34, -11],
    [-6, 7, 26, 31, 65, 56, 25, -20],
    [-14, 13, 6, 21, 23, 12, 17, -23],
    [-27, -2, -5, 12, 17, 6, 10, -25],
    [-26, -4, -4, -10, 3, 3, 33, -12],
    [-35, -1, -20, -23, -15, 24, 38, -22],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
PAWN_EG = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [178, 173, 158, 134, 147, 132, 165, 187],
    [94, 100, 85, 67, 56, 53, 82, 84],
    [32, 24, 13, 5, -2, 4, 17, 17],
    [13, 9, -3, -7, -7, -8, 3, -1],
    [4, 7, -6, 1, 0, -5, -1, -8],
    [13, 8, 8, 10, 13, 0, 2, -7],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
KNIGHT_MG = [
    [-167, -89, -34, -49, 61, -97, -15, -107],
    [-73, -41, 72, 36, 23, 62, 7, -17],
    [-47, 60, 37, 65, 84, 129, 73, 44],
    [-9, 17, 19, 53, 37, 69, 18, 22],
    [-13, 4, 16, 13, 28, 19, 21, -8],
    [-23, -9, 12, 10, 19, 17, 25, -16],
    [-29, -53, -12, -3, -1, 18, -14, -19],
    [-105, -21, -58, -33, -17, -28, -19, -23],
]
KNIGHT_EG = [
    [-58, -38, -13, -28, -31, -27, -63, -99],
    [-25, -8, -25, -2, -9, -25, -24, -52],
    [-24, -20, 10, 9, -1, -9, -19, -41],
    [-17, 3, 22, 22, 22, 11, 8, -18],
    [-18, -6, 16, 25, 16, 17, 4, -18],
    [-23, -3, -1, 15, 10, -3, -20, -22],
    [-42, -20, -10, -5, -2, -20, -23, -44],
    [-29, -51, -23, -15, -22, -18, -50, -64],
]
BISHOP_MG = [
    [-29, 4, -82, -37, -25, -42, 7, -8],
    [-26, 16, -18, -13, 30, 59, 18, -47],
    [-16, 37, 43, 40, 35, 50, 37, -2],
    [-4, 5, 19, 50, 37, 37, 7, -2],
    [-6, 13, 13, 26, 34, 12, 10, 4],
    [0, 15, 15, 15, 14, 27, 18, 10],
    [4, 15, 16, 0, 7, 21, 33, 1],
    [-33, -3, -14, -21, -13, -12, -39, -21],
]
BISHOP_EG = [
    [-14, -21, -11, -8, -7, -9, -17, -24],
    [-8, -4, 7, -12, -3, -13, -4, -14],
    [2, -8, 0, -1, -2, 6, 0, 4],
    [-3, 9, 12, 9, 14, 10, 3, 2],
    [-6, 3, 13, 19, 7, 10, -3, -9],
    [-12, -3, 8, 10, 13, 3, -7, -15],
    [-14, -18, -7, -1, 4, -9, -15, -27],
    [-23, -9, -23, -5, -9, -16, -5, -17],
]
ROOK_MG = [
    [32, 42, 32, 51, 63, 9, 31, 43],
    [27, 32, 58, 62, 80, 67, 26, 44],
    [-5, 19, 26, 36, 17, 45, 61, 16],
    [-24, -11, 7, 26, 24, 35, -8, -20],
    [-36, -26, -12, -1, 9, -7, 6, -23],
    [-45, -25, -16, -17, 3, 0, -5, -33],
    [-44, -16, -20, -9, -1, 11, -6, -71],
    [-19, -13, 1, 17, 16, 7, -37, -26],
]
ROOK_EG = [
    [13, 10, 18, 15, 12, 12, 8, 5],
    [11, 13, 13, 11, -3, 3, 8, 3],
    [7, 7, 7, 5, 4, -3, -5, -3],
    [4, 3, 13, 1, 2, 1, -1, 2],
    [3, 5, 8, 4, -5, -6, -8, -11],
    [-4, 0, -5, -1, -7, -12, -8, -16],
    [-6, -6, 0, 2, -9, -9, -11, -3],
    [-9, 2, 3, -1, -5, -13, 4, -20],
]
QUEEN_MG = [
    [-28, 0, 29, 12, 59, 44, 43, 45],
    [-24, -39, -5, 1, -16, 57, 28, 54],
    [-13, -17, 7, 8, 29, 56, 47, 57],
    [-27, -27, -16, -16, -1, 17, -2, 1],
    [-9, -26, -9, -10, -2, -4, 3, -3],
    [-14, 2, -11, -2, -5, 2, 14, 5],
    [-35, -8, 11, 2, 8, 15, -3, 1],
    [-1, -18, -9, 10, -15, -25, -31, -50],
]
QUEEN_EG = [
    [-9, 22, 22, 27, 27, 19, 10, 20],
    [-17, 20, 32, 41, 58, 25, 30, 0],
    [-20, 6, 9, 49, 47, 35, 19, 9],
    [3, 22, 24, 45, 57, 40, 57, 36],
    [-18, 28, 19, 47, 31, 34, 39, 23],
    [-16, -27, 15, 6, 9, 17, 10, 5],
    [-22, -23, -30, -16, -16, -23, -36, -32],
    [-33, -28, -22, -43, -5, -32, -20, -41],
]
KING_MG = [
    [-65, 23, 16, -15, -56, -34, 2, 13],
    [29, -1, -20, -7, -8, -4, -38, -29],
    [-9, 24, 2, -16, -20, 6, 22, -22],
    [-17, -20, -12, -27, -30, -25, -14, -36],
    [-49, -1, -27, -39, -46, -44, -33, -51],
    [-14, -14, -22, -46, -44, -30, -15, -27],
    [1, 7, -8, -64, -43, -16, 9, 8],
    [-15, 36, 12, -54, 8, -28, 24, 14],
]
KING_EG = [
    [-74, -35, -18, -18, -11, 15, 4, -17],
    [-12, 17, 14, 17, 17, 38, 23, 11],
    [10, 17, 23, 15, 20, 45, 44, 13],
    [-8, 22, 24, 27, 26, 33, 26, 3],
    [-18, -4, 21, 24, 27, 23, 9, -11],
    [-19, -3, 11, 21, 23, 16, 7, -9],
    [-27, -11, 4, 13, 14, 4, -5, -17],
    [-53, -34, -21, -11, -28, -14, -24, -43],
]


def _build(rows):
    """Rows are written rank 8 first; return a 64 entry table for White."""
    table = [0] * 64
    for r, row in enumerate(rows):
        rank = 7 - r
        for f, value in enumerate(row):
            table[chess.square(f, rank)] = value
    return table


# PST[color][piece_type][square], in midgame and endgame copies.
PST_MG = [[[0] * 64 for _ in range(7)] for _ in range(2)]
PST_EG = [[[0] * 64 for _ in range(7)] for _ in range(2)]


def _init_psts():
    sources = {
        chess.PAWN: (PAWN_MG, PAWN_EG),
        chess.KNIGHT: (KNIGHT_MG, KNIGHT_EG),
        chess.BISHOP: (BISHOP_MG, BISHOP_EG),
        chess.ROOK: (ROOK_MG, ROOK_EG),
        chess.QUEEN: (QUEEN_MG, QUEEN_EG),
        chess.KING: (KING_MG, KING_EG),
    }
    for piece_type, (mg_rows, eg_rows) in sources.items():
        mg = _build(mg_rows)
        eg = _build(eg_rows)
        for square in range(64):
            PST_MG[chess.WHITE][piece_type][square] = mg[square]
            PST_EG[chess.WHITE][piece_type][square] = eg[square]
            # Black is the vertical mirror, exactly as the 5x5 engine did it.
            mirrored = chess.square_mirror(square)
            PST_MG[chess.BLACK][piece_type][mirrored] = mg[square]
            PST_EG[chess.BLACK][piece_type][mirrored] = eg[square]


_init_psts()

# =============================================================================
# EVALUATION TERMS
# =============================================================================

BISHOP_PAIR_MG, BISHOP_PAIR_EG = 22, 48
DOUBLED_MG, DOUBLED_EG = -11, -24
ISOLATED_MG, ISOLATED_EG = -14, -18
PASSED_MG = [0, 5, 12, 22, 42, 78, 130, 0]
PASSED_EG = [0, 12, 24, 44, 78, 130, 200, 0]
ROOK_OPEN_MG, ROOK_OPEN_EG = 28, 12
ROOK_SEMI_MG, ROOK_SEMI_EG = 12, 6
TEMPO = 12

# King safety. The 5x5 board had nowhere to hide, so the original engine leaned
# on eval_king_mobility and a pawn screen. On 8x8 an attack is built over
# several moves by pieces converging on the squares around the king, and an
# evaluation that cannot see that coming will happily strip its own shield for
# a few centipawns of activity somewhere else.
KING_SHIELD_MG = 14
KING_OPEN_FILE_MG = -26  # no pawn of either colour on a file beside the king
KING_SEMI_FILE_MG = -13  # no pawn of ours, but one of theirs

# Attack units contributed per attacked square in the enemy king zone.
KING_ATTACK_WEIGHT = [0, 0, 20, 20, 40, 80, 0]
# Danger grows faster than linearly: two attackers are worth far more than one.
KING_DANGER = [min(600, (units * units) // 20) for units in range(128)]

# How far outside the caller's window pass 1 must land before pass 2 is
# skipped. This has to exceed everything the positional terms can contribute,
# or the search sees evaluations that are simply wrong. Measured over 1500
# random positions the positional swing was a median of 35cp, p99.9 of 385 and
# a maximum of 456, so 600 leaves real headroom. It is worth about 5-8% and,
# importantly, leaves the node count unchanged: at this margin the cut only
# fires where the full evaluation could not have changed the outcome. Lowering
# it to 240 looks 17% faster and is not - it searches MORE nodes, because the
# wrong scores degrade pruning.
LAZY_MARGIN = 600

# Quiescence discards captures whose SEE falls below this. Zero means "prune
# every capture that loses material at all", which is the textbook setting and
# is not obviously right here: at depth 5-6 the engine often cannot see the
# compensation for a sacrifice, so pruning the move prunes the tactic. A more
# negative value keeps speculative captures in the search.
QS_SEE_THRESHOLD = -110

MOBILITY_MG = [0, 0, 4, 4, 2, 1, 0]
MOBILITY_EG = [0, 0, 4, 3, 4, 2, 0]

# Every square ahead of `square` on its own file and the two adjacent files.
PASSED_MASK = [[0] * 64 for _ in range(2)]
# Every square strictly ahead on the same file, for doubled pawns.
FRONT_FILE = [[0] * 64 for _ in range(2)]
# The three squares in front of a king, for the pawn shield.
KING_SHIELD = [[0] * 64 for _ in range(2)]
# KING_ZONE[color][square]: the squares around a king of `color`, extended one
# rank towards the enemy, which is the direction an attack arrives from.
KING_ZONE = [[0] * 64 for _ in range(2)]
# The one to three file masks beside a king, precomputed so the eval does not
# redo square_file / range / clamping arithmetic at every node.
KING_FILES: list[tuple[int, ...]] = [()] * 64
ADJACENT_FILES = [0] * 8


def _init_masks():
    for f in range(8):
        mask = 0
        if f > 0:
            mask |= chess.BB_FILES[f - 1]
        if f < 7:
            mask |= chess.BB_FILES[f + 1]
        ADJACENT_FILES[f] = mask

    for square in range(64):
        f = chess.square_file(square)
        r = chess.square_rank(square)
        files = chess.BB_FILES[f] | ADJACENT_FILES[f]

        ahead_w = 0
        for rr in range(r + 1, 8):
            ahead_w |= chess.BB_RANKS[rr]
        ahead_b = 0
        for rr in range(0, r):
            ahead_b |= chess.BB_RANKS[rr]

        PASSED_MASK[chess.WHITE][square] = files & ahead_w
        PASSED_MASK[chess.BLACK][square] = files & ahead_b
        FRONT_FILE[chess.WHITE][square] = chess.BB_FILES[f] & ahead_w
        FRONT_FILE[chess.BLACK][square] = chess.BB_FILES[f] & ahead_b

        shield_w = 0
        shield_b = 0
        for df in (-1, 0, 1):
            nf = f + df
            if 0 <= nf <= 7:
                if r + 1 <= 7:
                    shield_w |= chess.BB_SQUARES[chess.square(nf, r + 1)]
                if r - 1 >= 0:
                    shield_b |= chess.BB_SQUARES[chess.square(nf, r - 1)]
        KING_SHIELD[chess.WHITE][square] = shield_w
        KING_SHIELD[chess.BLACK][square] = shield_b

        ring = chess.BB_KING_ATTACKS[square] | chess.BB_SQUARES[square]
        for color in (chess.WHITE, chess.BLACK):
            step = 1 if color == chess.WHITE else -1
            extended = ring
            for s in chess.scan_forward(ring):
                nr = chess.square_rank(s) + step
                if 0 <= nr <= 7:
                    extended |= chess.BB_SQUARES[
                        chess.square(chess.square_file(s), nr)]
            KING_ZONE[color][square] = extended

        KING_FILES[square] = tuple(
            chess.BB_FILES[nf]
            for nf in range(max(0, f - 1), min(7, f + 1) + 1)
        )


_init_masks()

# =============================================================================
# MOVE ORDERING TABLES
# =============================================================================

# MVV_LVA[attacker][victim], same shape as the 5x5 engine's table.
MVV_LVA = [[0] * 7 for _ in range(7)]
for _a in range(1, 7):
    for _v in range(1, 7):
        MVV_LVA[_a][_v] = PIECE_VALUE_MG[_v] * 16 - PIECE_VALUE_MG[_a]

HISTORY = [[[0] * 64 for _ in range(64)] for _ in range(7)]
# Killers/counters store packed moves (see pack_move below), not chess.Move
# objects: 0 means "none set" rather than a real move.
KILLERS = [[0, 0] for _ in range(MAX_PLY)]
COUNTER = [[0] * 64 for _ in range(64)]

# Continuation history learns which quiet move follows the prior one or two
# half-moves.  A 12-bit from/to move key is enough here because promotions are
# never quiet.  Two 4096 x 4096 int16 tables cost 64 MiB, small beside the TT
# while avoiding a hot-path Python dict.
CONTINUATION_HISTORY = np.zeros((2, 4096, 4096), dtype=np.int16)
CONTINUATION_MAX = 8192

# History aging, EXPERIMENTAL. Off by default - the shipped behavior is the
# flat, unbounded `+= depth * depth` at HISTORY's two update sites in
# negamax. When on, a cutoff-causing quiet move gets a bounded gravity bonus
# instead, and every OTHER quiet move already tried at that node before the
# cutoff gets a matching malus - the malus is what gives the scheme its
# "aging" property (a move that stops getting reinforced fades because other
# moves' mali pull it down, not just because its own growth is capped).
# Search a commit message near this comment for the arena result this was
# tested with before deciding whether it stays on.
HISTORY_AGING_ENABLED = False
HISTORY_MAX = 16384


def _history_bonus(depth):
    return min(depth * depth, HISTORY_MAX)


def _history_update(container, index, bonus):
    """Bounded gravity update: container[index] moves toward +/-HISTORY_MAX
    by `bonus`, rather than accumulating without bound. `bonus` may be
    negative (a malus)."""
    h = container[index]
    container[index] = h + bonus - h * abs(bonus) // HISTORY_MAX


def _move_history_index(move):
    """12-bit from/to key for quiet-move history tables."""
    return move.from_square | (move.to_square << 6)


def _continuation_update(prev_move, prev_prev_move, move, depth):
    """Reward a quiet cutoff in one- and two-ply continuation tables."""
    current = _move_history_index(move)
    bonus = min(depth * depth, CONTINUATION_MAX)
    for distance, earlier in enumerate((prev_move, prev_prev_move)):
        if earlier is None:
            continue
        row = CONTINUATION_HISTORY[distance, _move_history_index(earlier)]
        value = int(row[current])
        row[current] = value + bonus - value * bonus // CONTINUATION_MAX

# Late move reduction table, indexed [depth][move_index].
LMR = [[0] * 64 for _ in range(64)]
for _d in range(1, 64):
    for _i in range(1, 64):
        LMR[_d][_i] = int(0.75 + math.log(_d) * math.log(_i) / 2.25)

# =============================================================================
# MOVE ENCODING
# =============================================================================
# A chess.Move packed into a plain int: from_square (6 bits) | to_square
# (6 bits) << 6 | promotion (3 bits) << 12. 0 is reserved for "no move" - it
# is never produced by a real move, since a1a1 (from == to == 0) can't be
# legal, and the only other from==to==0 move, chess.Move.null(), is never
# stored anywhere a packed value is read back from (killers, counters, the TT
# move slot). Used so KILLERS/COUNTER/the TT can hold plain ints instead of
# chess.Move objects.


def pack_move(move: "chess.Move") -> int:
    return move.from_square | (move.to_square << 6) | ((move.promotion or 0) << 12)


def unpack_move(packed: int) -> "chess.Move":
    from_square = packed & 0x3F
    to_square = (packed >> 6) & 0x3F
    promotion = (packed >> 12) & 0x7
    return chess.Move(from_square, to_square, promotion or None)


# =============================================================================
# TRANSPOSITION TABLE
# =============================================================================
# Parallel typed arrays rather than a list of (key, depth, score, flag, move)
# tuples: array indexing and primitive dtypes at every probe/store, the hot
# path's most frequent operation, instead of Python-tuple unpacking and
# chess.Move objects. Bucketed by key & TT_MASK and replaced on depth, same
# scheme as before. TT_DEPTH doubles as the occupancy flag: -1 marks an empty
# bucket, since a real search depth is never negative.

TT_KEY = np.zeros(TT_SIZE, dtype=np.uint64)
TT_DEPTH = np.full(TT_SIZE, -1, dtype=np.int16)
TT_SCORE = np.zeros(TT_SIZE, dtype=np.int16)
TT_FLAG = np.zeros(TT_SIZE, dtype=np.uint8)
TT_MOVE = np.zeros(TT_SIZE, dtype=np.uint16)  # packed, see pack_move above


def tt_clear():
    TT_DEPTH.fill(-1)


# =============================================================================
# SEARCH STATE
# =============================================================================


class TimeUp(Exception):
    """Raised deep in the search to unwind to the last completed iteration."""


class Searcher:
    def __init__(self):
        self.hard_deadline = 0.0
        self.soft_deadline = 0.0
        self.nodes = 0
        self.qnodes = 0
        self.seldepth = 0
        self.path = []  # zobrist keys along the current search path
        self.game_keys = {}  # zobrist key -> times already played this game

        # Move-ordering telemetry, DEBUG-only (see search_root's summary
        # line). Purely additive counters at existing branch points - can't
        # change move selection, only report on it. cutoff_at_index[i] counts
        # beta cutoffs whose move was at ordered index i (3+ bucketed into
        # slot 3); tt_move_cutoffs is the subset from the staged TT-move-first
        # branch specifically, which never reaches the indexed loop at all.
        self.cutoff_at_index = [0, 0, 0, 0]
        self.tt_move_cutoffs = 0
        self.total_cutoffs = 0
        self.nmp_cutoffs = 0
        self.nmp_tries = 0
        self.lmr_full_researches = 0
        self.lmr_reductions = 0

    def reset_telemetry(self):
        self.cutoff_at_index = [0, 0, 0, 0]
        self.tt_move_cutoffs = 0
        self.total_cutoffs = 0
        self.nmp_cutoffs = 0
        self.nmp_tries = 0
        self.lmr_full_researches = 0
        self.lmr_reductions = 0

    def check_time(self, force=False):
        # Only touch the clock every 1024 nodes; perf_counter is not free.
        if (force or self.nodes & 1023 == 0) and time.perf_counter() >= self.hard_deadline:
            raise TimeUp


SEARCH = Searcher()


def _telemetry_snapshot():
    """A copy of SEARCH's 0b move-ordering counters, for taking a delta
    across a single root candidate's own subtree (see _root_window's trace
    hook below). Only ever called when a trace hook is in use - never in
    real play."""
    return {
        "cutoff_at_index": list(SEARCH.cutoff_at_index),
        "tt_move_cutoffs": SEARCH.tt_move_cutoffs,
        "total_cutoffs": SEARCH.total_cutoffs,
        "nmp_cutoffs": SEARCH.nmp_cutoffs,
        "nmp_tries": SEARCH.nmp_tries,
        "lmr_full_researches": SEARCH.lmr_full_researches,
        "lmr_reductions": SEARCH.lmr_reductions,
    }


def _telemetry_delta(before):
    after = _telemetry_snapshot()
    delta = {k: after[k] - before[k] for k in after if k != "cutoff_at_index"}
    delta["cutoff_at_index"] = [
        a - b for a, b in zip(after["cutoff_at_index"], before["cutoff_at_index"],
                               strict=True)
    ]
    return delta

# =============================================================================
# EVALUATION
# =============================================================================

# numpy mirrors of the PST/material tables above, for the jitted pass 1 below.
# Kept alongside the plain-Python tables rather than replacing them: pass 2
# and move ordering still read the originals, and duplicating a few small
# arrays is cheaper than threading numpy types through the whole file.
PST_MG_NP = np.array(PST_MG, dtype=np.int64)
PST_EG_NP = np.array(PST_EG, dtype=np.int64)
PIECE_VALUE_MG_NP = np.array(PIECE_VALUE_MG, dtype=np.int64)
PIECE_VALUE_EG_NP = np.array(PIECE_VALUE_EG, dtype=np.int64)
PHASE_WEIGHT_NP = np.array(PHASE_WEIGHT, dtype=np.int64)


@njit(cache=False)
def _jit_material_pst(bitboards, pst_mg, pst_eg, piece_value_mg, piece_value_eg,
                       phase_weight, bishop_pair_mg, bishop_pair_eg):
    """Material, PST and bishop pair, jitted. Mirrors evaluate()'s pass 1
    exactly; see the parity check referenced in the commit that added this
    before trusting it further than that.

    bitboards is a (2, 7) uint64 array, bitboards[color][piece_type], indices
    matching python-chess's own color (False/True) and piece-type (1..6)
    values so callers can build it directly from Board attributes.
    """
    mg = 0
    eg = 0
    phase = 0
    for color in range(2):
        sign = 1 if color == 1 else -1
        table_mg = pst_mg[color]
        table_eg = pst_eg[color]
        for piece_type in range(1, 7):
            bb = bitboards[color, piece_type]
            if bb == 0:
                continue

            count = 0
            sum_mg = 0
            sum_eg = 0
            piece_table_mg = table_mg[piece_type]
            piece_table_eg = table_eg[piece_type]
            remaining = bb
            while remaining:
                lsb = remaining & (-remaining)
                square = 0
                temp = lsb
                while temp > np.uint64(1):
                    temp >>= np.uint64(1)
                    square += 1
                sum_mg += piece_table_mg[square]
                sum_eg += piece_table_eg[square]
                count += 1
                remaining ^= lsb

            phase += phase_weight[piece_type] * count
            mg += sign * piece_value_mg[piece_type] * count
            eg += sign * piece_value_eg[piece_type] * count
            if piece_type == 3 and count >= 2:  # chess.BISHOP == 3
                mg += sign * bishop_pair_mg
                eg += sign * bishop_pair_eg
            mg += sign * sum_mg
            eg += sign * sum_eg

    return mg, eg, phase


def _load_nnue_model():
    """Load a quantized, project-trained evaluator from source or a zip.

    The model is deliberately a NumPy archive rather than an external ONNX
    file: `pkgutil.get_data()` reads it both from a normal checkout and the
    final zip submission, without extracting a mutable temporary file.  Its
    board-plane input is [color, piece type, square] and its hidden layer is
    evaluated from integer accumulators below.
    """
    try:
        payload = pkgutil.get_data(__name__, NNUE_RESOURCE)
    except Exception:
        # Deliberately unnarrowed: this loader's entire contract is "never
        # crash the engine over an optional resource". pkgutil.get_data()
        # calls importlib.util.find_spec() internally, which raises
        # ValueError (not OSError/TypeError) when __name__ == "__main__" -
        # exactly the self-test invocation (`uv run python agent.py`) and
        # plausibly close to how the platform's own loader might run this
        # file, so that specific case is not hypothetical. A second failure
        # mode here that crashes the whole module import over a missing
        # weights file would be strictly worse than the eval it's meant to
        # improve, so any exception at all falls through to the existing
        # hand-crafted evaluate() instead of raising.
        return None
    if payload is None:
        return None

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as model:
            feature_weights = np.asarray(model["feature_weights"], dtype=np.int16)
            hidden_bias = np.asarray(model["hidden_bias"], dtype=np.int32)
            output_weights = np.asarray(model["output_weights"], dtype=np.int16)
            output_bias = int(model["output_bias"])
            scale = int(model["scale"])
            blend = int(model["blend"])
    except (KeyError, OSError, ValueError, TypeError, EOFError):
        return None

    hidden_size = hidden_bias.shape[0] if hidden_bias.ndim == 1 else 0
    if (
        hidden_size == 0
        or feature_weights.shape != (2, 7, 64, hidden_size)
        or output_weights.shape != (hidden_size,)
        or scale <= 0
        or not 1 <= blend <= 100
    ):
        return None
    return feature_weights, hidden_bias, output_weights, output_bias, scale, blend


NNUE_MODEL = _load_nnue_model()


@njit(cache=False)
def _jit_nnue_score(bitboards, feature_weights, hidden_bias, output_weights,
                    output_bias, scale):
    """White-relative score from a quantized board-plane network."""
    hidden_size = hidden_bias.shape[0]
    hidden = hidden_bias.copy()
    one = np.uint64(1)
    for color in range(2):
        for piece_type in range(1, 7):
            remaining = bitboards[color, piece_type]
            while remaining:
                lsb = remaining & (-remaining)
                square = 0
                temp = lsb
                while temp > one:
                    temp >>= one
                    square += 1
                for hidden_index in range(hidden_size):
                    hidden[hidden_index] += feature_weights[
                        color, piece_type, square, hidden_index]
                remaining ^= lsb

    score = output_bias
    for hidden_index in range(hidden_size):
        activation = hidden[hidden_index]
        if activation > 0:
            score += activation * output_weights[hidden_index]
    return score // scale


def _nnue_score(bitboards):
    if NNUE_MODEL is None:
        return 0
    feature_weights, hidden_bias, output_weights, output_bias, scale, _blend = NNUE_MODEL
    return int(_jit_nnue_score(
        bitboards, feature_weights, hidden_bias, output_weights, output_bias, scale,
    ))


# Knight attacks are a fixed per-square pattern with no blockers to resolve,
# so the table is taken directly from python-chess rather than re-derived. The
# sliding tables below apply the same principle to bishops and rooks: each
# relevant blocker subset maps to its exact attack bitboard, addressed by a
# magic multiplier (BISHOP_MAGICS_NP/ROOK_MAGICS_NP below, found offline by
# brute force - not derived from any external engine, just a search over
# random sparse 64-bit values for one with no index collisions on the
# relevant blocker set). _init_sliding_attack_tables() below builds each
# table from python-chess's own attack dictionaries and then exhaustively
# re-verifies every entry against them before search begins, so a wrong or
# stale magic constant fails at import rather than silently misplaying.
KNIGHT_ATTACKS_NP = np.array([int(bb) for bb in chess.BB_KNIGHT_ATTACKS], dtype=np.uint64)
MOBILITY_MG_NP = np.array(MOBILITY_MG, dtype=np.int64)
MOBILITY_EG_NP = np.array(MOBILITY_EG, dtype=np.int64)
KING_ATTACK_WEIGHT_NP = np.array(KING_ATTACK_WEIGHT, dtype=np.int64)


BISHOP_MAGICS_NP = np.array([
    0x2084101400419202, 0x1004040802043000, 0x810A102000280,
    0x508060040040000, 0x1023104000208800, 0x8201040A40800090,
    0x4100809410408001, 0x8801008801011106, 0x2001510C08008404,
    0x42104240C81, 0x101104400902000, 0x130080841000204,
    0x1A8040460001028, 0x20510088102, 0x108A10104400,
    0x1014120084140A08, 0x200006A02C0104, 0x22101008900100C0,
    0x100120424143C0, 0x4149804802004040, 0x1842001402110160,
    0x6022000104818C00, 0x8142000420824820, 0x2080082841100,
    0x3024400904104400, 0x8200004090204, 0xC0280210005040,
    0x4000C80000820040, 0x4181001015004002, 0x8008004202030090,
    0x4108A40421000, 0x4112000410402, 0xC0108402C201400,
    0x8043000244100, 0x8100414800404A05, 0x200220180280380,
    0x1040102020A20080, 0x10004440020502, 0x2008202410822,
    0x2204411A0060A080, 0xA2718080488C002, 0x1623880888080201,
    0xD008150006100, 0x1292018000100, 0x100200840810,
    0x82081020200101, 0x2200A00A200091B, 0x100C02862A001040,
    0x424208210100004, 0x48829411604241, 0x5420804208241621,
    0x3082002042020242, 0x3000446004640801, 0x40811020010B0020,
    0x4840A80214D20040, 0x20080109006000, 0x804422014000,
    0xD80AB18021020, 0x80A0691044445006, 0x20A808,
    0x210010E0A201, 0x618010A0016300, 0xA3400E08210920,
    0x92001040C2080,
], dtype=np.uint64)

ROOK_MAGICS_NP = np.array([
    0x1080004000102080, 0x440051000402000, 0x800982E0001000,
    0x4C80080210008004, 0x1200102A00042009, 0x880118002000C00,
    0x400040810128302, 0x80008001304100, 0x1108800020844000,
    0x98400050026000, 0x8022002200401080, 0x2902000910204201,
    0x1241001802050010, 0x4000804200805400, 0x801000100020004,
    0x8202000412028043, 0x28000400144E004, 0x8040050023008448,
    0x460008020100080, 0x42120021420018, 0x1222818004000801,
    0x10080104402010, 0x2440040082052850, 0x86000A492484,
    0x4140048880006C41, 0x4802001C0035008, 0x20110100402000,
    0x8010002100110018, 0x4104008080062800, 0x2010340080020080,
    0x1244D0080800200, 0x40010003000884C2, 0x804000800A20,
    0x3220086004101, 0x2A00301004010, 0x400801000800800,
    0x1000080080805400, 0x210200050A001008, 0x4810810C004608,
    0x200006082000401, 0x4004A0988000, 0x408201020024,
    0x10002001808012, 0x120041218A0010, 0xC000040801010011,
    0x2001000C00030008, 0x20020110240008, 0x4040080C5020004,
    0x104800028400080, 0x4000400484A00380, 0xB201040820200,
    0x200900008210100, 0x901000800100700, 0x8002011428100200,
    0x21070810120400, 0x10090402428200, 0x62024080910062,
    0x8024104208210082, 0x200082001011241, 0x8210041000190121,
    0x80810050062C0801, 0x2002002801041022, 0x4410002000C01C1,
    0x80200401208900C2,
], dtype=np.uint64)


def _magic_index(occupied, mask, magic, shift):
    return (((occupied & mask) * magic) & ((1 << 64) - 1)) >> shift


def _init_sliding_attack_tables():
    bishop_masks = [int(mask) for mask in chess.BB_DIAG_MASKS]
    rook_masks = [
        int(chess.BB_FILE_MASKS[square] | chess.BB_RANK_MASKS[square])
        for square in range(64)
    ]
    bishop_table = np.zeros(
        (64, 1 << max(mask.bit_count() for mask in bishop_masks)), dtype=np.uint64,
    )
    rook_table = np.zeros(
        (64, 1 << max(mask.bit_count() for mask in rook_masks)), dtype=np.uint64,
    )
    bishop_shifts = [64 - mask.bit_count() for mask in bishop_masks]
    rook_shifts = [64 - mask.bit_count() for mask in rook_masks]

    for square, mask in enumerate(bishop_masks):
        for occupied, attacks in chess.BB_DIAG_ATTACKS[square].items():
            index = _magic_index(
                int(occupied), mask, int(BISHOP_MAGICS_NP[square]), bishop_shifts[square],
            )
            bishop_table[square, index] = attacks

    for square, mask in enumerate(rook_masks):
        for rank_occupied, rank_attacks in chess.BB_RANK_ATTACKS[square].items():
            for file_occupied, file_attacks in chess.BB_FILE_ATTACKS[square].items():
                occupied = int(rank_occupied | file_occupied)
                attacks = int(rank_attacks | file_attacks)
                index = _magic_index(
                    occupied, mask, int(ROOK_MAGICS_NP[square]), rook_shifts[square],
                )
                rook_table[square, index] = attacks

    # Every possible relevant blocker pattern is represented exactly once.
    for square, mask in enumerate(bishop_masks):
        for occupied, attacks in chess.BB_DIAG_ATTACKS[square].items():
            index = _magic_index(
                int(occupied), mask, int(BISHOP_MAGICS_NP[square]), bishop_shifts[square],
            )
            assert int(bishop_table[square, index]) == attacks
    for square, mask in enumerate(rook_masks):
        for rank_occupied, rank_attacks in chess.BB_RANK_ATTACKS[square].items():
            for file_occupied, file_attacks in chess.BB_FILE_ATTACKS[square].items():
                occupied = int(rank_occupied | file_occupied)
                attacks = int(rank_attacks | file_attacks)
                index = _magic_index(
                    occupied, mask, int(ROOK_MAGICS_NP[square]), rook_shifts[square],
                )
                assert int(rook_table[square, index]) == attacks

    return (
        np.array(bishop_masks, dtype=np.uint64), bishop_table,
        np.array(bishop_shifts, dtype=np.uint8), np.array(rook_masks, dtype=np.uint64),
        rook_table, np.array(rook_shifts, dtype=np.uint8),
    )


(BISHOP_MASKS_NP, BISHOP_ATTACKS_NP, BISHOP_SHIFTS_NP,
 ROOK_MASKS_NP, ROOK_ATTACKS_NP, ROOK_SHIFTS_NP) = _init_sliding_attack_tables()


@njit(cache=False)
def _slider_attacks_jit(square, occupied, masks, magics, shifts, attacks):
    # Public helpers can be called with a signed Python int; normalize before
    # multiplication so magic indexing always wraps as uint64.
    blockers = np.uint64(occupied) & masks[square]
    index = (blockers * magics[square]) >> shifts[square]
    return attacks[square, int(index)]


@njit(cache=False)
def _diag_attacks_jit(square, occupied):
    return _slider_attacks_jit(
        square, occupied, BISHOP_MASKS_NP, BISHOP_MAGICS_NP, BISHOP_SHIFTS_NP,
        BISHOP_ATTACKS_NP,
    )


@njit(cache=False)
def _orth_attacks_jit(square, occupied):
    return _slider_attacks_jit(
        square, occupied, ROOK_MASKS_NP, ROOK_MAGICS_NP, ROOK_SHIFTS_NP,
        ROOK_ATTACKS_NP,
    )


@njit(cache=False)
def _jit_mobility(bitboards, occupied, own_occ, target_zone,
                   mobility_mg, mobility_eg, king_attack_weight, knight_attacks):
    """Mobility and king-zone-attack contribution, jitted. Mirrors evaluate()'s
    mobility loop exactly: for every knight/bishop/rook/queen, how many squares
    it attacks that it doesn't already occupy (mobility) and how much of the
    target king zone it covers (attack_units), by color.
    """
    mg = 0
    eg = 0
    au_black = 0
    au_white = 0
    one = np.uint64(1)

    for color in range(2):
        sign = 1 if color == 1 else -1
        own = own_occ[color]
        zone = target_zone[color]

        for piece_type in range(2, 6):  # KNIGHT=2 .. QUEEN=5
            bb = bitboards[color, piece_type]
            if bb == 0:
                continue
            w_mg = mobility_mg[piece_type]
            w_eg = mobility_eg[piece_type]
            atk_weight = king_attack_weight[piece_type]

            remaining = bb
            while remaining:
                lsb = remaining & (-remaining)
                square = 0
                temp = lsb
                while temp > one:
                    temp >>= one
                    square += 1

                if piece_type == 2:
                    attacked = knight_attacks[square]
                elif piece_type == 3:
                    attacked = _diag_attacks_jit(square, occupied)
                elif piece_type == 4:
                    attacked = _orth_attacks_jit(square, occupied)
                else:
                    attacked = (_diag_attacks_jit(square, occupied)
                                | _orth_attacks_jit(square, occupied))

                moves_bb = attacked & ~own
                moves = 0
                mv = moves_bb
                while mv:
                    mv &= mv - one
                    moves += 1
                mg += sign * w_mg * moves
                eg += sign * w_eg * moves

                if zone != 0:
                    hits_bb = attacked & zone
                    hits = 0
                    hv = hits_bb
                    while hv:
                        hv &= hv - one
                        hits += 1
                    if hits:
                        if color == 0:
                            au_black += atk_weight * hits
                        else:
                            au_white += atk_weight * hits

                remaining ^= lsb

    return mg, eg, au_black, au_white


def _piece_bitboards(board):
    """(2, 7) uint64 array, bitboards[color][piece_type]: colors matching
    int(chess.BLACK)=0 / int(chess.WHITE)=1, piece types matching
    chess.PAWN..chess.KING (1..6). Shared by evaluate() and see() - both feed
    a board into jitted code in this shape, and building it by hand twice is
    exactly the kind of place a transcription bug likes to hide (a bare
    True/False used as a numpy index is boolean masking, not integer
    indexing, and raises a confusingly-shaped IndexError - see below).
    """
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    bitboards = np.empty((2, 7), dtype=np.uint64)
    black_row = bitboards[0]
    white_row = bitboards[1]
    black_row[chess.PAWN] = board.pawns & black
    white_row[chess.PAWN] = board.pawns & white
    black_row[chess.KNIGHT] = board.knights & black
    white_row[chess.KNIGHT] = board.knights & white
    black_row[chess.BISHOP] = board.bishops & black
    white_row[chess.BISHOP] = board.bishops & white
    black_row[chess.ROOK] = board.rooks & black
    white_row[chess.ROOK] = board.rooks & white
    black_row[chess.QUEEN] = board.queens & black
    white_row[chess.QUEEN] = board.queens & white
    black_row[chess.KING] = board.kings & black
    white_row[chess.KING] = board.kings & white
    return bitboards


def _material_pst_phase(board):
    """Fresh (mg, eg, phase): the raw, unclamped pass-1 output evaluate()'s
    jitted material/PST/bishop-pair pass always computed inline, factored out
    so search_root/_root_window can seed the incremental mg_eg_phase value
    threaded through negamax/qsearch (see the INCREMENTAL EVALUATION section
    below) without duplicating this block a second time.
    """
    bitboards = _piece_bitboards(board)
    mg, eg, phase = _jit_material_pst(
        bitboards, PST_MG_NP, PST_EG_NP, PIECE_VALUE_MG_NP, PIECE_VALUE_EG_NP,
        PHASE_WEIGHT_NP, BISHOP_PAIR_MG, BISHOP_PAIR_EG,
    )
    return int(mg), int(eg), int(phase)


def evaluate(board, alpha=-INFINITY, beta=INFINITY, mg_eg_phase=None):
    """Static evaluation in centipawns, from the side to move's point of view.

    Two passes. The first is material, piece-square tables and the bishop pair,
    which is cheap and accounts for nearly all of the score in nearly every
    position. If that alone already sits a full LAZY_MARGIN outside the window
    the caller cares about, no amount of positional detail will drag it back
    in, so the second pass is skipped. Quiescence makes almost every evaluate
    call in the search and cuts on stand-pat most of the time, which is exactly
    the case this is for.

    Passing no window forces the full evaluation.

    mg_eg_phase, when given, is the pass-1 output already maintained
    incrementally by the caller (see _eval_delta / _child_eval below) - pass
    1 then skips _material_pst_phase's recompute entirely. Every call site
    that does not thread it still gets a correct pass 1, just a slower one:
    same invariant as the incremental Zobrist key.
    """
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_pawns = board.pawns & white
    black_pawns = board.pawns & black
    # Needed by pass 2's mobility jit below whenever pass 2 runs at all, so
    # built unconditionally regardless of whether pass 1 uses the incremental
    # mg_eg_phase or recomputes fresh.
    bitboards = _piece_bitboards(board)

    # ---- pass 1: material, PST, bishop pair --------------------------------
    if mg_eg_phase is not None and not EVAL_ASSERT:
        mg, eg, phase = mg_eg_phase
    else:
        mg, eg, phase = _jit_material_pst(
            bitboards, PST_MG_NP, PST_EG_NP, PIECE_VALUE_MG_NP, PIECE_VALUE_EG_NP,
            PHASE_WEIGHT_NP, BISHOP_PAIR_MG, BISHOP_PAIR_EG,
        )
        mg, eg, phase = int(mg), int(eg), int(phase)
        if EVAL_ASSERT and mg_eg_phase is not None:
            assert (mg, eg, phase) == mg_eg_phase, (
                board.fen(), (mg, eg, phase), mg_eg_phase,
            )

    if phase > MAX_PHASE:
        phase = MAX_PHASE

    white_to_move = board.turn == chess.WHITE
    lazy = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    lazy += TEMPO if white_to_move else -TEMPO
    if not white_to_move:
        lazy = -lazy

    # A loaded neural model can move the full score either way, so the
    # handcrafted pass-1 bound is only sound when no model is active.
    if (NNUE_MODEL is None
            and (lazy - LAZY_MARGIN >= beta or lazy + LAZY_MARGIN <= alpha)):
        return lazy

    # ---- pass 2: the positional terms -------------------------------------
    # Zone around each king, held as the zone its OPPONENT is attacking.
    # Skipped when the attacker has no queen: a king hunt without one almost
    # never breaks through, and dropping the term there costs nothing and buys
    # back most of what king safety spends in queenless positions.
    kings = board.kings
    queens = board.queens
    white_king_bb = kings & white
    black_king_bb = kings & black
    zone_of = [0, 0]
    if black_king_bb and queens & white:
        zone_of[chess.WHITE] = KING_ZONE[chess.BLACK][black_king_bb.bit_length() - 1]
    if white_king_bb and queens & black:
        zone_of[chess.BLACK] = KING_ZONE[chess.WHITE][white_king_bb.bit_length() - 1]
    attack_units = [0, 0]

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color else -1
        own = white if color else black
        own_pawns = white_pawns if color else black_pawns
        their_pawns = black_pawns if color else white_pawns

        for square in chess.scan_forward(own_pawns):
            # Doubled: another friendly pawn ahead on the same file.
            if FRONT_FILE[color][square] & own_pawns:
                mg += sign * DOUBLED_MG
                eg += sign * DOUBLED_EG
            # Isolated: no friendly pawn on either adjacent file.
            if not (ADJACENT_FILES[chess.square_file(square)] & own_pawns):
                mg += sign * ISOLATED_MG
                eg += sign * ISOLATED_EG
            # Passed: no enemy pawn anywhere in the forward span.
            if not (PASSED_MASK[color][square] & their_pawns):
                rank = chess.square_rank(square)
                advance = rank if color else 7 - rank
                mg += sign * PASSED_MG[advance]
                eg += sign * PASSED_EG[advance]

        for square in chess.scan_forward(board.rooks & own):
            file_bb = chess.BB_FILES[chess.square_file(square)]
            if not (file_bb & own_pawns):
                if file_bb & their_pawns:
                    mg += sign * ROOK_SEMI_MG
                    eg += sign * ROOK_SEMI_EG
                else:
                    mg += sign * ROOK_OPEN_MG
                    eg += sign * ROOK_OPEN_EG

        for square in chess.scan_forward(kings & own):
            shield = (KING_SHIELD[color][square] & own_pawns).bit_count()
            mg += sign * KING_SHIELD_MG * shield
            # An open or half-open file beside the king is the road an attack
            # travels down. Pushing the pawn in front of your own king should
            # cost more than the tempo it buys.
            for file_bb in KING_FILES[square]:
                if not (file_bb & own_pawns):
                    mg += sign * (
                        KING_SEMI_FILE_MG if file_bb & their_pawns
                        else KING_OPEN_FILE_MG
                    )

    # Mobility and king-zone attacks (jitted): how many squares each
    # knight/bishop/rook/queen attacks that it doesn't already occupy, and how
    # much of the opponent's king zone it covers. Handles both colors in one
    # call, so it runs once here rather than inside the per-color loop above.
    own_occ = np.array([black, white], dtype=np.uint64)
    target_zone_arr = np.array([zone_of[0], zone_of[1]], dtype=np.uint64)
    occupied_bb = np.uint64(board.occupied)
    mob_mg, mob_eg, au_black, au_white = _jit_mobility(
        bitboards, occupied_bb, own_occ, target_zone_arr,
        MOBILITY_MG_NP, MOBILITY_EG_NP, KING_ATTACK_WEIGHT_NP, KNIGHT_ATTACKS_NP,
    )
    mg += int(mob_mg)
    eg += int(mob_eg)
    attack_units[chess.BLACK] = int(au_black)
    attack_units[chess.WHITE] = int(au_white)

    # King danger is a midgame term only: with the queens off, a king in the
    # open is an asset, not a liability, and the endgame PSTs already say so.
    mg += (KING_DANGER[min(attack_units[chess.WHITE] // 8, 127)]
           - KING_DANGER[min(attack_units[chess.BLACK] // 8, 127)])

    # Taper between the midgame and endgame scores.
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    score += TEMPO if white_to_move else -TEMPO

    if NNUE_MODEL is not None:
        blend = NNUE_MODEL[-1]
        neural_score = _nnue_score(bitboards)
        score = (score * (100 - blend) + neural_score * blend) // 100

    return score if white_to_move else -score


def game_phase(board):
    phase = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        phase += PHASE_WEIGHT[piece_type] * (
            board.pieces_mask(piece_type, chess.WHITE).bit_count()
            + board.pieces_mask(piece_type, chess.BLACK).bit_count()
        )
    return min(phase, MAX_PHASE)


# =============================================================================
# MOVE ORDERING
# =============================================================================


def order_moves(board, moves, tt_move, ply, prev_move, prev_prev_move=None):
    """TT move, then captures by MVV-LVA, then promotions, then killers, then
    the counter move, history and continuation history. Alpha-beta only pays
    when this is right."""
    them = board.occupied_co[not board.turn]
    killer0, killer1 = KILLERS[ply] if ply < MAX_PLY else (0, 0)
    counter = 0
    if prev_move is not None:
        counter = COUNTER[prev_move.from_square][prev_move.to_square]

    scored: list[tuple[int, chess.Move]] = []
    append = scored.append
    for move in moves:
        if move == tt_move:
            append((1 << 30, move))
            continue

        to_square = move.to_square
        victim_bb = chess.BB_SQUARES[to_square] & them

        if move.promotion:
            score = 900_000_000 + PIECE_VALUE_MG[move.promotion]
            if victim_bb:
                score += MVV_LVA[chess.PAWN][board.piece_type_at(to_square)]
        elif victim_bb:
            attacker = board.piece_type_at(move.from_square)
            victim = board.piece_type_at(to_square)
            # Winning and equal captures stay ahead of everything quiet.
            # Losing ones drop below killers and history: they are still worth
            # searching, just not first.
            if see(board, move) < 0:
                score = 100_000_000 + MVV_LVA[attacker][victim]
            else:
                score = 800_000_000 + MVV_LVA[attacker][victim]
        elif board.is_en_passant(move):
            score = 800_000_000 + MVV_LVA[chess.PAWN][chess.PAWN]
        else:
            packed = pack_move(move)
            if packed == killer0:
                score = 700_000_000
            elif packed == killer1:
                score = 690_000_000
            elif counter and packed == counter:
                score = 680_000_000
            else:
                piece_type = board.piece_type_at(move.from_square)
                score = HISTORY[piece_type][move.from_square][to_square]
                current = _move_history_index(move)
                if prev_move is not None:
                    score += int(CONTINUATION_HISTORY[
                        0, _move_history_index(prev_move), current])
                if prev_prev_move is not None:
                    score += int(CONTINUATION_HISTORY[
                        1, _move_history_index(prev_prev_move), current])

        append((score, move))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [move for _, move in scored]


def order_captures(board, moves):
    them = board.occupied_co[not board.turn]
    scored: list[tuple[int, chess.Move]] = []
    for move in moves:
        to_square = move.to_square
        if chess.BB_SQUARES[to_square] & them:
            attacker = board.piece_type_at(move.from_square)
            victim = board.piece_type_at(to_square)
            score = MVV_LVA[attacker][victim]
        else:
            score = MVV_LVA[chess.PAWN][chess.PAWN]  # en passant
        if move.promotion:
            score += PIECE_VALUE_MG[move.promotion] * 16
        scored.append((score, move))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [move for _, move in scored]


# =============================================================================
# STATIC EXCHANGE EVALUATION
# =============================================================================
# What MVV-LVA cannot tell you: whether a capture actually WINS material once
# the recapture sequence plays out. MVV-LVA sorts NxQ above everything, even
# when the queen is defended three times and the knight is lost for nothing.
# SEE plays the exchange out on one square, always recapturing with the
# cheapest available attacker, and returns the material balance.
#
# The 5x5 engine leaned on _cap_order alone. On 8x8 there are far more sliding
# pieces and far more batteries behind them, so "is this capture safe" stops
# being obvious from the two pieces involved.

SEE_VALUE = [0, 100, 325, 335, 500, 975, 20000]
SEE_VALUE_NP = np.array(SEE_VALUE, dtype=np.int64)

# A board holds at most 32 pieces, so an exchange sequence on one square can
# never involve more captures than that; 40 leaves headroom without
# meaning anything more precise than "clearly enough".
SEE_MAX_DEPTH = 40

# PAWN_ATTACKS_NP[color][square]: squares a pawn of `color` standing on
# `square` attacks. Taken directly from python-chess's own BB_PAWN_ATTACKS
# (indexed [chess.BLACK, chess.WHITE] = [0, 1], matching every other color
# index in this file) rather than re-derived - one less place for a
# transcription bug to live, same reasoning as KNIGHT_ATTACKS_NP above.
PAWN_ATTACKS_NP = np.array(
    [[int(bb) for bb in chess.BB_PAWN_ATTACKS[color]]
     for color in (chess.BLACK, chess.WHITE)],
    dtype=np.uint64,
)
KING_ATTACKS_NP = np.array([int(bb) for bb in chess.BB_KING_ATTACKS], dtype=np.uint64)


@njit(cache=False)
def _jit_attackers_to(target, occupied, white_pawns, black_pawns, all_knights,
                       all_kings, all_bishops_queens, all_rooks_queens,
                       pawn_attacks, king_attacks, knight_attacks):
    """Combined white+black attacker bitboard of `target` under `occupied`.
    Mirrors board.attackers_mask(WHITE, ...) | board.attackers_mask(BLACK,
    ...) - including its final "& occupied", which is what lets a piece
    whose square has been vacated (captured earlier in the same exchange)
    drop out regardless of piece type, not just for sliding pieces.
    """
    attackers = pawn_attacks[0, target] & white_pawns
    attackers |= pawn_attacks[1, target] & black_pawns
    attackers |= knight_attacks[target] & all_knights
    attackers |= king_attacks[target] & all_kings
    attackers |= _diag_attacks_jit(target, occupied) & all_bishops_queens
    attackers |= _orth_attacks_jit(target, occupied) & all_rooks_queens
    return attackers & occupied


@njit(cache=False)
def _jit_least_valuable(attackers, bitboards, side):
    """Cheapest attacker of `side` in `attackers`, as (square, piece_type), or
    (-1, 0) if none. Ties (two attackers of the same type) resolve to the
    lowest-indexed square rather than _least_valuable's old highest-indexed
    one; harmless, since same-type attackers carry the same SEE value and the
    other one stays in `attackers` for the next iteration either way - see
    the Phase B parity check before trusting that claim further than this
    comment does.
    """
    one = np.uint64(1)
    for piece_type in range(1, 7):
        subset = attackers & bitboards[side, piece_type]
        if subset:
            lsb = subset & (-subset)
            square = 0
            temp = lsb
            while temp > one:
                temp >>= one
                square += 1
            return square, piece_type
    return -1, 0


@njit(cache=False)
def _jit_see(bitboards, occupied, from_square, target, side_to_move, captured,
             attacker, ep_square, see_value, pawn_attacks, king_attacks,
             knight_attacks):
    """Numba twin of the exchange loop see() used to run in pure Python.
    side_to_move is the mover's color (1=white, 0=black, matching bitboards'
    own color index). ep_square is the extra pawn square to vacate for an en
    passant capture, or -1. No early exit - see the note this carries over
    from the pre-jit version: the classic SEE pruning test is unsound the way
    this loop is folded back, and stopping early once mis-scored a battery
    position by 160cp.
    """
    one = np.uint64(1)
    gain = np.zeros(SEE_MAX_DEPTH, dtype=np.int64)
    gain[0] = see_value[captured] if captured else 0

    occ = occupied & ~(one << np.uint64(from_square))
    if ep_square >= 0:
        occ &= ~(one << np.uint64(ep_square))

    white_pawns = bitboards[1, 1]
    black_pawns = bitboards[0, 1]
    all_knights = bitboards[0, 2] | bitboards[1, 2]
    all_bishops_queens = bitboards[0, 3] | bitboards[1, 3] | bitboards[0, 5] | bitboards[1, 5]
    all_rooks_queens = bitboards[0, 4] | bitboards[1, 4] | bitboards[0, 5] | bitboards[1, 5]
    all_kings = bitboards[0, 6] | bitboards[1, 6]

    attackers = _jit_attackers_to(
        target, occ, white_pawns, black_pawns, all_knights, all_kings,
        all_bishops_queens, all_rooks_queens, pawn_attacks, king_attacks,
        knight_attacks)

    side = 1 - side_to_move
    on_square = attacker
    depth = 0

    while depth < SEE_MAX_DEPTH - 1:
        square, piece_type = _jit_least_valuable(attackers, bitboards, side)
        if square < 0:
            break

        depth += 1
        # Standing on the square is worth what we just took, minus whatever
        # the opponent can win back by continuing.
        gain[depth] = see_value[on_square] - gain[depth - 1]

        on_square = piece_type
        occ &= ~(one << np.uint64(square))
        attackers = _jit_attackers_to(
            target, occ, white_pawns, black_pawns, all_knights, all_kings,
            all_bishops_queens, all_rooks_queens, pawn_attacks, king_attacks,
            knight_attacks)
        side = 1 - side

    # Fold back: at each step the side to move may decline the capture.
    while depth:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1

    return gain[0]


def see(board, move):
    """Material won by `move` in centipawns, assuming best play by both sides.

    Negative means the exchange loses material. Promotions and en passant are
    approximated: the point is ordering and pruning, not exactness.

    Pins are ignored, because attackers_mask does not know about them, so a
    defender that cannot legally recapture is still counted. Checked against a
    brute-force minimax of the exchange over 1083 captures from 300 random
    positions: 2 disagreements, both pins. That is the usual trade and the
    usual error rate; resolving pins here would cost more than it saves.

    The exchange loop itself runs jitted (_jit_see) - this wrapper only
    builds the primitive inputs it needs from `board`.
    """
    target = move.to_square
    from_square = move.from_square

    captured = board.piece_type_at(target)
    if captured is None:
        captured = chess.PAWN if board.is_en_passant(move) else 0

    attacker = board.piece_type_at(from_square)
    if attacker is None:
        return 0

    ep_square = -1
    if board.is_en_passant(move):
        ep_square = target + (-8 if board.turn == chess.WHITE else 8)

    bitboards = _piece_bitboards(board)
    side_to_move = 1 if board.turn else 0

    return int(_jit_see(
        bitboards, np.uint64(board.occupied), from_square, target,
        side_to_move, captured, attacker, ep_square, SEE_VALUE_NP,
        PAWN_ATTACKS_NP, KING_ATTACKS_NP, KNIGHT_ATTACKS_NP,
    ))


# =============================================================================
# INCREMENTAL ZOBRIST
# =============================================================================
# chess.polyglot.zobrist_hash recomputes the full board every call - ~8% of
# runtime per the profile. The piece-square component is the part worth
# updating incrementally (an O(pieces) loop collapsed to an O(1) XOR); the
# castling and en-passant components are cheap fixed-cost checks already, so
# those are recomputed via python-chess's own ZobristHasher methods before and
# after each push rather than hand-derived, which is both simpler and removes
# an entire class of risk (an early draft of the castling square mapping for
# this got the rook-origin bit indices backwards against python-chess's own
# square numbering - reusing the library's own logic for that part avoids
# needing to get it right a second time by hand).
#
# key=None on negamax is the fallback: recompute from scratch, exactly as
# before this existed. Every call site that does not thread an incremental
# key through still gets a correct key, just not a faster one - there is no
# path that produces a wrong key silently, only a slower-but-correct one.
ZOBRIST_ASSERT = False  # debug only: recomputes and asserts on every push

_ZOBRIST_HASHER = chess.polyglot.ZobristHasher(chess.polyglot.POLYGLOT_RANDOM_ARRAY)
_ZOBRIST_TURN_KEY = chess.polyglot.POLYGLOT_RANDOM_ARRAY[780]

# PIECE_SQUARE_KEY[pivot][piece_type][square], pivot = int(color) matching
# enumerate(board.occupied_co) in chess.polyglot.ZobristHasher.hash_board
# (0 = black, 1 = white) exactly, so no color-index adjustment is needed
# anywhere this table is read.
PIECE_SQUARE_KEY = [[[0] * 64 for _ in range(7)] for _ in range(2)]
for _pivot in (0, 1):
    for _piece_type in range(1, 7):
        _piece_index = (_piece_type - 1) * 2 + _pivot
        for _square in range(64):
            PIECE_SQUARE_KEY[_pivot][_piece_type][_square] = (
                chess.polyglot.POLYGLOT_RANDOM_ARRAY[64 * _piece_index + _square]
            )


def _piece_square_delta(board, move):
    """XOR delta for the piece-square component of the hash, for `move` not
    yet pushed onto `board`. Handles captures, en passant, castling's rook
    move, and promotion. A null move (from_square == to_square) moves
    nothing, by construction of chess.Move.null().
    """
    if move.from_square == move.to_square:
        return 0

    pivot = 1 if board.turn else 0
    from_sq = move.from_square
    to_sq = move.to_square
    piece_type = board.piece_type_at(from_sq)

    delta = PIECE_SQUARE_KEY[pivot][piece_type][from_sq]
    delta ^= PIECE_SQUARE_KEY[pivot][move.promotion or piece_type][to_sq]

    if board.is_en_passant(move):
        captured_sq = to_sq + (-8 if board.turn else 8)
        delta ^= PIECE_SQUARE_KEY[1 - pivot][chess.PAWN][captured_sq]
    else:
        captured_type = board.piece_type_at(to_sq)
        if captured_type is not None:
            delta ^= PIECE_SQUARE_KEY[1 - pivot][captured_type][to_sq]

    if board.is_castling(move):
        queenside = chess.square_file(to_sq) < chess.square_file(from_sq)
        rank = chess.A1 if board.turn else chess.A8
        rook_from = rank if queenside else rank + 7
        rook_to = (rank + 3) if queenside else (rank + 5)
        delta ^= PIECE_SQUARE_KEY[pivot][chess.ROOK][rook_from]
        delta ^= PIECE_SQUARE_KEY[pivot][chess.ROOK][rook_to]

    return delta


def _zobrist_pre(board, move):
    """Snapshot the state that must be read before `move` is pushed. Pair
    with _zobrist_post after board.push(move)."""
    return (
        _piece_square_delta(board, move),
        _ZOBRIST_HASHER.hash_castling(board) ^ _ZOBRIST_HASHER.hash_ep_square(board),
    )


def _zobrist_post(board, key, pre):
    """Combine a _zobrist_pre snapshot with post-push state to get the
    pushed position's key. Call immediately after board.push(move)."""
    delta, old_extra = pre
    new_extra = (
        _ZOBRIST_HASHER.hash_castling(board) ^ _ZOBRIST_HASHER.hash_ep_square(board)
    )
    new_key = key ^ delta ^ old_extra ^ new_extra ^ _ZOBRIST_TURN_KEY
    if ZOBRIST_ASSERT:
        assert new_key == chess.polyglot.zobrist_hash(board), (
            board.fen(), new_key, chess.polyglot.zobrist_hash(board)
        )
    return new_key


# =============================================================================
# INCREMENTAL EVALUATION
# =============================================================================
# evaluate()'s pass 1 (material, PST, bishop pair) recomputes from scratch on
# every call, and qsearch alone calls evaluate() at nearly every leaf. This
# maintains that pass's (mg, eg, phase) incrementally through push/pop instead,
# the same shape as the incremental Zobrist key above: a delta computed from
# `board` before a move is pushed, folded into the running total, threaded as
# an optional argument through negamax/qsearch/_root_window. A call site that
# does not thread it still gets a correct pass 1 out of evaluate() - just a
# slower, freshly recomputed one - never a wrong one; see evaluate()'s own
# docstring for that invariant stated the same way it already is for the key.
#
# mg/eg/phase here are the same *raw, unclamped* values _material_pst_phase
# returns - the MAX_PHASE clamp is applied in evaluate() itself, identically
# whether that pass-1 output arrived incrementally or fresh.
EVAL_ASSERT = False  # debug only: recomputes and asserts on every push


def _eval_delta(board, move):
    """(mg_delta, eg_delta, phase_delta) for `move`, not yet pushed onto
    `board`. A structural twin of _piece_square_delta above, with the
    material/phase/bishop-pair components _piece_square_delta doesn't need
    (Zobrist has no positional-bonus term) added on top of the same
    piece-square shape: a piece leaves from_square and arrives at to_square
    (possibly promoted), a captured or en-passant-captured piece is removed,
    and castling also moves the rook. A null move (from_square == to_square)
    changes nothing.

    Unlike _piece_square_delta this needs no post-push state, so there is no
    separate _eval_post: everything it needs is readable before the push.
    """
    if move.from_square == move.to_square:
        return 0, 0, 0

    from_sq = move.from_square
    to_sq = move.to_square
    color = board.turn  # the mover
    sign = 1 if color else -1
    piece_type = board.piece_type_at(from_sq)
    new_type = move.promotion or piece_type

    mg = sign * (PST_MG[color][new_type][to_sq] - PST_MG[color][piece_type][from_sq])
    eg = sign * (PST_EG[color][new_type][to_sq] - PST_EG[color][piece_type][from_sq])
    phase = 0

    if move.promotion:
        mg += sign * (PIECE_VALUE_MG[new_type] - PIECE_VALUE_MG[chess.PAWN])
        eg += sign * (PIECE_VALUE_EG[new_type] - PIECE_VALUE_EG[chess.PAWN])
        phase += PHASE_WEIGHT[new_type] - PHASE_WEIGHT[chess.PAWN]
        if new_type == chess.BISHOP:
            # A second bishop turns the pair bonus on; a first or third+
            # doesn't change it.
            our_bishops = (board.bishops & board.occupied_co[color]).bit_count()
            if our_bishops == 1:
                mg += sign * BISHOP_PAIR_MG
                eg += sign * BISHOP_PAIR_EG

    if board.is_en_passant(move):
        captured_sq = to_sq + (-8 if color else 8)
        their_sign = -sign
        mg -= their_sign * (PST_MG[not color][chess.PAWN][captured_sq]
                             + PIECE_VALUE_MG[chess.PAWN])
        eg -= their_sign * (PST_EG[not color][chess.PAWN][captured_sq]
                             + PIECE_VALUE_EG[chess.PAWN])
        # A pawn's phase weight is 0 and it can't be one half of a bishop
        # pair, so neither needs handling here.
    else:
        captured_type = board.piece_type_at(to_sq)
        if captured_type is not None:
            their_color = not color
            their_sign = -sign
            mg -= their_sign * (PST_MG[their_color][captured_type][to_sq]
                                 + PIECE_VALUE_MG[captured_type])
            eg -= their_sign * (PST_EG[their_color][captured_type][to_sq]
                                 + PIECE_VALUE_EG[captured_type])
            phase -= PHASE_WEIGHT[captured_type]
            if captured_type == chess.BISHOP:
                # Dropping from exactly 2 bishops to 1 turns the pair bonus
                # off; dropping from 3+ to 2+, or from 1 to 0, doesn't.
                their_bishops = (
                    board.bishops & board.occupied_co[their_color]).bit_count()
                if their_bishops == 2:
                    mg -= their_sign * BISHOP_PAIR_MG
                    eg -= their_sign * BISHOP_PAIR_EG

    if board.is_castling(move):
        queenside = chess.square_file(to_sq) < chess.square_file(from_sq)
        rank = chess.A1 if color else chess.A8
        rook_from = rank if queenside else rank + 7
        rook_to = (rank + 3) if queenside else (rank + 5)
        mg += sign * (PST_MG[color][chess.ROOK][rook_to]
                      - PST_MG[color][chess.ROOK][rook_from])
        eg += sign * (PST_EG[color][chess.ROOK][rook_to]
                      - PST_EG[color][chess.ROOK][rook_from])

    return mg, eg, phase


def _child_eval(board, move, mg_eg_phase):
    """mg_eg_phase for the position after `move`, not yet pushed onto
    `board`, or None if the caller has no incremental value to propagate -
    evaluate() then falls back to a full recompute, as it always did before
    this existed. Call before board.push(move), like _zobrist_pre."""
    if mg_eg_phase is None:
        return None
    mg_delta, eg_delta, phase_delta = _eval_delta(board, move)
    return (
        mg_eg_phase[0] + mg_delta,
        mg_eg_phase[1] + eg_delta,
        mg_eg_phase[2] + phase_delta,
    )


# =============================================================================
# QUIESCENCE
# =============================================================================

DELTA_MARGIN = 200


def qsearch(board, alpha, beta, ply, mg_eg_phase=None):
    SEARCH.nodes += 1
    SEARCH.qnodes += 1
    SEARCH.check_time()

    if ply > SEARCH.seldepth:
        SEARCH.seldepth = ply
    if ply >= MAX_PLY - 1:
        return evaluate(board, mg_eg_phase=mg_eg_phase)

    in_check = board.is_check()
    key = None

    if in_check:
        # Never stand pat in check. Search every evasion so we do not call a
        # position quiet while the king is attacked. That search is a forced
        # sequence of evasions and intermezzo checks, which can repeat or run
        # into the 50-move rule the same way a line in negamax can — negamax
        # guards against that, but qsearch otherwise never does.
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0
        key = chess.polyglot.zobrist_hash(board)
        if key in SEARCH.path or SEARCH.game_keys.get(key, 0) >= 2:
            return 0
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply
        moves = order_moves(board, moves, None, ply, None)
        best = -INFINITY
    else:
        stand_pat = evaluate(board, alpha, beta, mg_eg_phase)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        best = stand_pat
        captures = list(board.generate_legal_captures())
        # A quiet leaf with no captures left can still be a stalemate rather
        # than the live position the material-based stand-pat assumes. Real
        # stalemates are an endgame phenomenon, so gate the legality scan
        # (unlike captures, not cheap to generate) behind a low piece count
        # instead of paying it on every quiet leaf.
        if not captures and board.occupied.bit_count() <= 8 and not any(board.legal_moves):
            return 0
        moves = order_captures(board, captures)

    if key is not None:
        SEARCH.path.append(key)
    try:
        for move in moves:
            if not in_check:
                # Delta pruning: if winning this piece outright still leaves us
                # far below alpha, the capture cannot rescue the position.
                victim = board.piece_type_at(move.to_square)
                gain = PIECE_VALUE_MG[victim] if victim else PIECE_VALUE_MG[chess.PAWN]
                if move.promotion:
                    gain += PIECE_VALUE_MG[move.promotion]
                if best + gain + DELTA_MARGIN < alpha:
                    continue

                # SEE pruning. MVV-LVA happily puts NxQ first even when the
                # queen is defended three times; playing the exchange out says
                # whether the capture actually wins material. A capture that
                # loses material cannot raise alpha, and quiescence is where
                # most of the nodes are.
                if see(board, move) < QS_SEE_THRESHOLD:
                    continue

            child_mg_eg_phase = _child_eval(board, move, mg_eg_phase)
            board.push(move)
            score = -qsearch(board, -beta, -alpha, ply + 1, child_mg_eg_phase)
            board.pop()

            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
    finally:
        if key is not None:
            SEARCH.path.pop()

    return best


# =============================================================================
# NEGAMAX (PVS)
# =============================================================================

FUTILITY_MARGIN = [0, 100, 180, 280, 400, 540, 700]
RFP_MARGIN = 75
LMP_COUNT = [0, 5, 8, 13, 20, 29, 40]

# Singular extension, added under a live tournament deadline without a 40-game
# arena to confirm it - see CLAUDE.md's Open Work item 4 and the round-62
# investigation this responds to (a critical move that scores fine through a
# shallow search and only reveals a forced win several plies deeper). Minimal
# and heavily guarded on purpose: only a TT move from a deep, non-upper-bound
# entry is ever tested, only at nodes already deep enough that the
# verification search's cost (a full move-list generation plus a reduced
# search over every alternative) stays rare. SINGULAR_EXT_ENABLED = False is
# an instant rollback to the prior, arena-confirmed-adjacent behavior.
SINGULAR_EXT_ENABLED = True
SINGULAR_MIN_DEPTH = 8       # verification search only pays off when rare
SINGULAR_MARGIN = 60         # cp the TT move must clear every alternative by


def negamax(board, depth, alpha, beta, ply, prev_move, allow_null=True, key=None,
            mg_eg_phase=None, excluded_move=None, prev_prev_move=None):
    SEARCH.nodes += 1
    SEARCH.check_time()

    is_pv = beta - alpha > 1
    original_alpha = alpha

    if key is None:
        key = chess.polyglot.zobrist_hash(board)

    # --- draws -------------------------------------------------------------
    if ply > 0:
        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0
        # A repeat inside the current line is a draw. So is a position the real
        # game has already seen TWICE, since arriving again makes three and the
        # referee claims threefold automatically. Counting matters: treating a
        # single earlier occurrence as a draw blinded the search completely,
        # returning 0 for every root move whose child had been played before.
        if key in SEARCH.path or SEARCH.game_keys.get(key, 0) >= 2:
            return 0

    if ply >= MAX_PLY - 1:
        return evaluate(board, mg_eg_phase=mg_eg_phase)

    in_check = board.is_check()

    # Check extension, capped the same way the 5x5 engine capped it. Applied
    # before the TT probe below so a cached entry is checked against the depth
    # this node actually intends to search. Probing with the pre-extension
    # depth while storing with the post-extension depth let an entry one ply
    # shallower than the extension guarantees satisfy the cutoff — exactly the
    # positions the extension exists to search deeper, undermined by the TT.
    if in_check and ply < 40:
        depth += 1

    # --- transposition probe ----------------------------------------------
    index = key & TT_MASK
    tt_depth = int(TT_DEPTH[index])
    tt_hit = tt_depth >= 0 and int(TT_KEY[index]) == key
    tt_move = None

    if tt_hit:
        packed_tt_move = int(TT_MOVE[index])
        tt_move = unpack_move(packed_tt_move) if packed_tt_move else None
        if not is_pv and tt_depth >= depth and ply > 0 and excluded_move is None:
            score = int(TT_SCORE[index])
            # Mate scores are stored relative to the root; re-anchor them.
            if score > MATE_BOUND:
                score -= ply
            elif score < -MATE_BOUND:
                score += ply
            flag = int(TT_FLAG[index])
            if flag == TT_EXACT:
                return score
            if flag == TT_LOWER and score >= beta:
                return score
            if flag == TT_UPPER and score <= alpha:
                return score

    if depth <= 0:
        return qsearch(board, alpha, beta, ply, mg_eg_phase)

    static_eval = -MATE_SCORE if in_check else evaluate(board, mg_eg_phase=mg_eg_phase)

    # --- whole node pruning ------------------------------------------------
    if not is_pv and not in_check and abs(beta) < MATE_BOUND:
        # Reverse futility: so far above beta that shedding a few centipawns
        # per remaining ply still cannot drag us below it.
        if depth <= 6 and static_eval - RFP_MARGIN * depth >= beta:
            return static_eval

        # Null move pruning. R = 2 + depth/6 here; the 5x5 engine used R = 1
        # because a 25 square board leaves almost no zugzwang-free slack.
        if allow_null and depth >= 3 and static_eval >= beta:
            has_pieces = (
                board.occupied_co[board.turn] & ~board.pawns & ~board.kings
            )
            if has_pieces:
                reduction = 2 + depth // 6
                SEARCH.nmp_tries += 1
                board.push(chess.Move.null())
                SEARCH.path.append(key)
                try:
                    # A null move changes nothing material/PST/phase-wise
                    # (see _eval_delta's own null-move case), so the current
                    # mg_eg_phase carries straight through unchanged.
                    score = -negamax(board, depth - reduction - 1, -beta,
                                     -beta + 1, ply + 1, None, allow_null=False,
                                     mg_eg_phase=mg_eg_phase)
                finally:
                    SEARCH.path.pop()
                    board.pop()
                if score >= beta:
                    SEARCH.nmp_cutoffs += 1
                    return beta if score > MATE_BOUND else score

    # --- internal iterative deepening -------------------------------------
    if tt_move is None and depth >= 4 and is_pv:
        negamax(board, depth - 2, alpha, beta, ply, prev_move, allow_null, key,
                mg_eg_phase, prev_prev_move=prev_prev_move)
        tt_depth = int(TT_DEPTH[index])
        tt_hit = tt_depth >= 0 and int(TT_KEY[index]) == key
        if tt_hit:
            packed_tt_move = int(TT_MOVE[index])
            tt_move = unpack_move(packed_tt_move) if packed_tt_move else None

    them = board.occupied_co[not board.turn]
    can_futility = (
        not is_pv
        and not in_check
        and depth <= 6
        and abs(alpha) < MATE_BOUND
        and static_eval + FUTILITY_MARGIN[depth] <= alpha
    )
    can_lmp = not is_pv and not in_check and depth <= 6 and abs(alpha) < MATE_BOUND

    best_score = -INFINITY
    best_move = None
    searched = 0

    # Staged move generation. A TT move is always searched first when present
    # (order_moves gives it absolute priority below), and python-chess's own
    # legal-move generation is comparatively expensive. Board.is_legal() does
    # not build that list, so if the TT move alone is legal and already
    # produces a cutoff, every other move was going to be skipped anyway —
    # try it before paying for the full list.
    tt_move_legal = tt_move is not None and board.is_legal(tt_move)
    cutoff = False
    # Quiet moves tried at this node so far, for HISTORY_AGING_ENABLED's
    # malus: only ever appended to (or read back) when that flag is on, so
    # it costs nothing beyond an empty list when it's off.
    tried_quiets: list[tuple[int, int, int]] = []

    # Singular extension probe. excluded_move is None here for every ordinary
    # node (this only ever runs one level deep - the verification call below
    # passes excluded_move=tt_move, and that call's own guard on
    # excluded_move is None then blocks a nested probe at the same position).
    # If no other legal move can even reach within SINGULAR_MARGIN of the TT
    # move's own score at half depth, the TT move is trusted as uniquely good
    # here and searched one ply deeper than normal.
    se_extension = 0
    if (SINGULAR_EXT_ENABLED and excluded_move is None and ply > 0
            and depth >= SINGULAR_MIN_DEPTH and tt_move_legal
            and tt_hit and tt_depth >= depth - 3
            and int(TT_FLAG[index]) != TT_UPPER):
        tt_score = int(TT_SCORE[index])
        if tt_score > MATE_BOUND:
            tt_score -= ply
        elif tt_score < -MATE_BOUND:
            tt_score += ply
        if abs(tt_score) < MATE_BOUND:
            singular_beta = tt_score - SINGULAR_MARGIN
            verify_depth = (depth - 1) // 2
            se_score = negamax(board, verify_depth, singular_beta - 1,
                                singular_beta, ply, prev_move,
                                allow_null=allow_null, key=key,
                                mg_eg_phase=mg_eg_phase, excluded_move=tt_move,
                                prev_prev_move=prev_prev_move)
            if se_score < singular_beta:
                se_extension = 1

    SEARCH.path.append(key)
    try:
        if tt_move_legal and tt_move is not None and tt_move != excluded_move:
            move = tt_move
            is_capture = bool(chess.BB_SQUARES[move.to_square] & them)
            is_quiet = not is_capture and not move.promotion

            zpre = _zobrist_pre(board, move)
            child_mg_eg_phase = _child_eval(board, move, mg_eg_phase)
            board.push(move)
            child_key = _zobrist_post(board, key, zpre)
            score = -negamax(board, depth - 1 + se_extension, -beta, -alpha,
                             ply + 1, move, key=child_key,
                             mg_eg_phase=child_mg_eg_phase,
                             prev_prev_move=prev_move)
            piece_type = board.piece_type_at(move.to_square)
            board.pop()
            searched = 1

            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    cutoff = True
                    SEARCH.total_cutoffs += 1
                    SEARCH.tt_move_cutoffs += 1
                    if is_quiet:
                        packed = pack_move(move)
                        if ply < MAX_PLY:
                            slot = KILLERS[ply]
                            if slot[0] != packed:
                                slot[1] = slot[0]
                                slot[0] = packed
                        if prev_move is not None:
                            COUNTER[prev_move.from_square][
                                prev_move.to_square] = packed
                        if piece_type:
                            if HISTORY_AGING_ENABLED:
                                _history_update(
                                    HISTORY[piece_type][move.from_square],
                                    move.to_square, _history_bonus(depth))
                            else:
                                HISTORY[piece_type][move.from_square][
                                    move.to_square] += depth * depth
                        _continuation_update(prev_move, prev_prev_move, move, depth)
            if not cutoff and HISTORY_AGING_ENABLED and is_quiet and piece_type:
                # Tried, didn't cut off: a malus candidate if a later move in
                # the main loop below does.
                tried_quiets.append((piece_type, move.from_square, move.to_square))

        if not cutoff:
            moves = list(board.legal_moves)
            if not moves:
                # Checkmate, or stalemate. Stalemate was a LOSS in the 5x5
                # variant; in standard chess it is a draw and must score 0.
                return -MATE_SCORE + ply if in_check else 0

            moves = order_moves(board, moves, tt_move, ply, prev_move, prev_prev_move)
            # Forced-reply extension: one legal move costs nothing extra to
            # search one ply deeper, since there is nothing else at this node
            # to spend that node budget on. Deliberately the strict len==1
            # case only, not "a few" - that version risks over-extending
            # zugzwang endgames where a side has 2-3 legal moves that are all
            # bad, not forced. Targets the same failure family as rounds 46
            # and 62 (a long forcing sequence the search was a ply too shallow
            # to see through), which is exactly where a real forced reply
            # (check evasions, escaping a mating net) shows up as this.
            one_reply_ext = 1 if len(moves) == 1 else 0

            for move_index, move in enumerate(moves):
                if tt_move_legal and move == tt_move:
                    continue  # already searched above

                is_capture = bool(chess.BB_SQUARES[move.to_square] & them)
                is_quiet = not is_capture and not move.promotion

                if is_quiet and searched > 0:
                    # Late move pruning: in a node that already looks bad, stop
                    # looking at quiet moves once enough have failed.
                    if can_lmp and move_index >= LMP_COUNT[depth]:
                        continue
                    # Futility: a quiet move cannot lift a static score that is
                    # already a full margin below alpha.
                    if can_futility:
                        continue

                zpre = _zobrist_pre(board, move)
                child_mg_eg_phase = _child_eval(board, move, mg_eg_phase)
                board.push(move)
                child_key = _zobrist_post(board, key, zpre)
                gives_check = board.is_check()

                if searched == 0:
                    score = -negamax(board, depth - 1 + one_reply_ext, -beta,
                                     -alpha, ply + 1, move, key=child_key,
                                     mg_eg_phase=child_mg_eg_phase,
                                     prev_prev_move=prev_move)
                else:
                    reduction = 0
                    if is_quiet and depth >= 3 and not gives_check and move_index >= 3:
                        reduction = LMR[min(depth, 63)][min(move_index, 63)]
                        if is_pv:
                            reduction -= 1
                        if reduction < 0:
                            reduction = 0
                        if reduction > depth - 2:
                            reduction = depth - 2

                    score = -negamax(board, depth - 1 - reduction, -alpha - 1,
                                     -alpha, ply + 1, move, key=child_key,
                                     mg_eg_phase=child_mg_eg_phase,
                                     prev_prev_move=prev_move)

                    # The reduced search beat alpha, so redo it at full depth.
                    if reduction > 0:
                        SEARCH.lmr_reductions += 1
                        if score > alpha:
                            SEARCH.lmr_full_researches += 1
                            score = -negamax(board, depth - 1, -alpha - 1, -alpha,
                                             ply + 1, move, key=child_key,
                                             mg_eg_phase=child_mg_eg_phase,
                                             prev_prev_move=prev_move)

                    # PVS re-search on a full window inside the bounds.
                    if alpha < score < beta:
                        score = -negamax(board, depth - 1, -beta, -alpha,
                                         ply + 1, move, key=child_key,
                                         mg_eg_phase=child_mg_eg_phase,
                                         prev_prev_move=prev_move)

                piece_type = board.piece_type_at(move.to_square)
                board.pop()
                searched += 1

                if score > best_score:
                    best_score = score
                    best_move = move
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            SEARCH.total_cutoffs += 1
                            SEARCH.cutoff_at_index[min(move_index, 3)] += 1
                            if is_quiet:
                                # Killer, counter move and history updates,
                                # lifted from the 5x5 engine's ordering scheme.
                                packed = pack_move(move)
                                if ply < MAX_PLY:
                                    slot = KILLERS[ply]
                                    if slot[0] != packed:
                                        slot[1] = slot[0]
                                        slot[0] = packed
                                if prev_move is not None:
                                    COUNTER[prev_move.from_square][
                                        prev_move.to_square] = packed
                                if piece_type:
                                    if HISTORY_AGING_ENABLED:
                                        bonus = _history_bonus(depth)
                                        _history_update(
                                            HISTORY[piece_type][move.from_square],
                                            move.to_square, bonus)
                                        malus = -bonus
                                        for m_pt, m_from, m_to in tried_quiets:
                                            _history_update(
                                                HISTORY[m_pt][m_from], m_to, malus)
                                    else:
                                        HISTORY[piece_type][move.from_square][
                                            move.to_square] += depth * depth
                                _continuation_update(
                                    prev_move, prev_prev_move, move, depth)
                            break
                if HISTORY_AGING_ENABLED and is_quiet and piece_type:
                    # Tried, didn't cut off this node: a malus candidate if a
                    # later move does.
                    tried_quiets.append((piece_type, move.from_square, move.to_square))
    finally:
        SEARCH.path.pop()

    if best_score == -INFINITY:
        # Everything was pruned. Return a static score rather than handing
        # -INFINITY back to the parent.
        return static_eval
    assert best_move is not None  # some move set it, or best_score is -INFINITY

    # --- transposition store ----------------------------------------------
    if best_score <= original_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    stored = best_score
    if stored > MATE_BOUND:
        stored += ply
    elif stored < -MATE_BOUND:
        stored -= ply

    if not tt_hit or depth >= tt_depth:
        TT_KEY[index] = key
        TT_DEPTH[index] = depth
        TT_SCORE[index] = stored
        TT_FLAG[index] = flag
        TT_MOVE[index] = pack_move(best_move)

    return best_score


# =============================================================================
# ROOT SEARCH
# =============================================================================


def _root_window(board, root_moves, best_move, depth, alpha, beta, base_stack,
                  trace=None):
    """Search every root move once inside (alpha, beta).

    Returns (move, score, finished, failed_high). A score at or below alpha is
    a fail low and a score at or above beta a fail high; either means the window
    was wrong and the caller has to widen it and come back.

    trace, when given, is called once per candidate as trace(depth, index,
    move, info) - info is a dict with score (None if TimeUp hit before this
    candidate's search finished), nodes, elapsed_s, completed, is_first (this
    candidate went first in move order - the closest thing to "was it the
    root TT/PV move" at this level), and a telemetry delta (0b's SEARCH
    counters, scoped to just this candidate's own subtree). Every real-play
    call site leaves this None - analysis-only, and verified inert when unset
    with the same fixed-depth parity technique used for every other optional
    argument in this file (mg_eg_phase, key, ...).
    """
    iter_move = None
    iter_score = -INFINITY
    ordered = order_moves(board, root_moves, best_move, 0, None)
    root_key = chess.polyglot.zobrist_hash(board)
    root_mg_eg_phase = _material_pst_phase(board)

    for index, move in enumerate(ordered):
        zpre = _zobrist_pre(board, move)
        child_mg_eg_phase = _child_eval(board, move, root_mg_eg_phase)
        if trace is not None:
            nodes_before = SEARCH.nodes
            trace_t0 = time.perf_counter()
            tel_before = _telemetry_snapshot()
        board.push(move)
        child_key = _zobrist_post(board, root_key, zpre)
        timed_out = False
        score = None
        try:
            if index == 0:
                score = -negamax(board, depth - 1, -beta, -alpha, 1, move,
                                 key=child_key, mg_eg_phase=child_mg_eg_phase)
            else:
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, 1, move,
                                 key=child_key, mg_eg_phase=child_mg_eg_phase)
                if alpha < score < beta:
                    score = -negamax(board, depth - 1, -beta, -alpha, 1, move,
                                     key=child_key, mg_eg_phase=child_mg_eg_phase)
        except TimeUp:
            timed_out = True
        finally:
            # Unwind to our own depth rather than popping once. When TimeUp is
            # raised deep in the tree it propagates straight through the move
            # loops in negamax and qsearch, which push before the recursive
            # call and pop after it, so every frame in flight leaks its push.
            # Popping once would leave the board several plies down, and every
            # legality check and san() after this point would then be answering
            # about the wrong position.
            while len(board.move_stack) > base_stack:
                board.pop()
            if trace is not None:
                info = {
                    "score": score,
                    "nodes": SEARCH.nodes - nodes_before,
                    "elapsed_s": time.perf_counter() - trace_t0,
                    "completed": not timed_out,
                    "is_first": index == 0,
                    "telemetry": _telemetry_delta(tel_before),
                }
                trace(depth, index, move, info)

        if timed_out:
            return iter_move, iter_score, False, False
        assert score is not None  # not timed_out means the try block completed

        if score > iter_score:
            iter_score = score
            iter_move = move
            if score >= beta:
                # Fail high. The score is only a lower bound, so searching the
                # rest of the list under a window we already know is wrong buys
                # nothing; widen and start again.
                return iter_move, iter_score, True, True
            if score > alpha:
                alpha = score

    return iter_move, iter_score, True, False


def search_root(board, soft_s, hard_s, trace=None):
    """Iterative deepening. Returns (move, score, depth, pv).

    trace is passed straight through to _root_window - see that function's
    docstring. None at the real get_move() call site; analysis-only.
    """
    start = time.perf_counter()
    SEARCH.hard_deadline = start + hard_s
    SEARCH.soft_deadline = start + soft_s
    SEARCH.nodes = 0
    SEARCH.qnodes = 0
    SEARCH.seldepth = 0
    SEARCH.path = []
    SEARCH.reset_telemetry()

    base_stack = len(board.move_stack)

    root_moves = list(board.legal_moves)
    if not root_moves:
        return None, 0, 0, ""

    best_move = root_moves[0]
    best_score = 0
    completed_depth = 0
    pv = ""
    prev_score = None
    prev_move = None
    last_iteration_nodes = 0

    # The root loop is explicit rather than deferred to negamax, so best_move
    # always comes from a search we actually ran. Reading it back out of the TT
    # instead is unsafe: buckets are indexed by key & TT_MASK, so a colliding
    # position can evict the root entry and leave a stale move behind.
    for depth in range(1, MAX_DEPTH_CAP + 1):
        iteration_start_nodes = SEARCH.nodes

        # Aspiration window. After a few plies the score barely moves between
        # iterations, so searching the full width every time pays for a range
        # the answer almost never uses. Guess narrow around the last score and
        # widen only when the guess turns out wrong.
        if (depth < ASPIRATION_MIN_DEPTH or prev_score is None
                or abs(prev_score) > MATE_BOUND):
            alpha, beta = -INFINITY, INFINITY
            delta = 0
        else:
            delta = ASPIRATION_DELTA
            alpha, beta = prev_score - delta, prev_score + delta

        # An aspiration fail returns a bound, not a usable iteration. Keep the
        # previous completed result until a re-search closes the window.
        completed_iteration = False
        finished = True
        iter_move = None
        while True:
            try:
                # Recursive search checks the clock every 1024 nodes. The
                # re-search loop has no node work of its own, so check it on
                # every pass to avoid starting a fresh window after expiry.
                SEARCH.check_time(force=True)
            except TimeUp:
                finished = False
                break

            iter_move, iter_score, finished, failed_high = _root_window(
                board, root_moves, best_move, depth, alpha, beta, base_stack,
                trace=trace)
            if not finished:
                break
            if delta and failed_high:
                delta *= 2
                beta = min(INFINITY, iter_score + delta)
                continue
            if delta and iter_score <= alpha:
                # Fail low: iter_score is an upper bound, so it cannot replace
                # the prior iteration until the widened search completes.
                delta *= 2
                alpha = max(-INFINITY, iter_score - delta)
                continue
            completed_iteration = True
            break

        if delta == 0 and not completed_iteration and iter_move is not None:
            # A plain full-window search never returns a mere bound (that
            # ambiguity only exists once alpha/beta have been narrowed by
            # aspiration), so a TimeUp partway through _root_window still
            # leaves a genuine, exact comparison among whatever candidates
            # were fully searched before the clock ran out - restores the
            # pre-aspiration "a partial iteration is still usable" behavior
            # for exactly the case that carries no bound-vs-exact risk. An
            # actual aspiration attempt (delta != 0) still requires
            # completed_iteration below, since that path can genuinely
            # return only a bound - the bug this rewrite exists to fix.
            best_move = iter_move
            best_score = iter_score

        if not completed_iteration or not finished:
            break

        assert iter_move is not None
        best_move = iter_move
        best_score = iter_score

        completed_depth = depth
        pv = _extract_pv(board, best_move, 8)

        # Effective branching factor: what this ply cost against the last one.
        # Perfect ordering in chess approaches sqrt(35), about 6, so a figure
        # near 5 says ordering is healthy and one near 9 says it is not. Logged
        # every move so rated games report it without any extra tooling.
        nodes_before = last_iteration_nodes
        last_iteration_nodes = SEARCH.nodes - iteration_start_nodes
        ebf = last_iteration_nodes / nodes_before if nodes_before else 0.0

        if DEBUG:
            elapsed = time.perf_counter() - start
            nps = int(SEARCH.nodes / elapsed) if elapsed > 0 else 0
            print(
                f"  depth {depth:2d} seldepth {SEARCH.seldepth:2d} "
                f"score {_format_score(best_score)} nodes {SEARCH.nodes} "
                f"ebf {ebf:4.1f} nps {nps} time {elapsed * 1000:.0f}ms pv {pv}",
                file=sys.stderr,
            )

        # A forced mate is found; no deeper search can improve on it.
        if abs(best_score) > MATE_BOUND:
            break

        # Fail-low extension. Stopping on schedule is right when the score is
        # steady, and wrong exactly when it is not: a root score falling ply
        # over ply means the last iteration was lying to us and the move we are
        # about to play rests on it. Buy the extra ply instead. Same, less
        # urgently, when the best move is still changing.
        budget = soft_s
        if depth >= 4 and prev_score is not None:
            if best_score < prev_score - 40:
                budget = soft_s * 2.2
            elif best_move != prev_move:
                budget = soft_s * 1.5
        # Stay clear of the hard limit: reaching it costs a whole iteration's
        # work, since a search aborted mid-iteration contributes nothing past
        # the moves it already finished.
        budget = min(budget, hard_s * 0.85)

        prev_score = best_score
        prev_move = best_move

        # Do not start an iteration we almost certainly cannot finish; each
        # extra ply costs roughly 2-3x the last one.
        if time.perf_counter() - start > budget * 0.5:
            break

    if DEBUG:
        total_cutoffs = SEARCH.total_cutoffs
        if total_cutoffs:
            idx_pct = [100.0 * c / total_cutoffs for c in SEARCH.cutoff_at_index]
            tt_pct = 100.0 * SEARCH.tt_move_cutoffs / total_cutoffs
        else:
            idx_pct = [0.0, 0.0, 0.0, 0.0]
            tt_pct = 0.0
        nmp_pct = (100.0 * SEARCH.nmp_cutoffs / SEARCH.nmp_tries
                   if SEARCH.nmp_tries else 0.0)
        lmr_pct = (100.0 * SEARCH.lmr_full_researches / SEARCH.lmr_reductions
                   if SEARCH.lmr_reductions else 0.0)
        print(
            f"  ordering: cutoffs tt {tt_pct:4.1f}% idx0 {idx_pct[0]:4.1f}% "
            f"idx1 {idx_pct[1]:4.1f}% idx2 {idx_pct[2]:4.1f}% idx3+ {idx_pct[3]:4.1f}% "
            f"| nmp {nmp_pct:4.1f}% ({SEARCH.nmp_cutoffs}/{SEARCH.nmp_tries}) "
            f"| lmr re-search {lmr_pct:4.1f}% "
            f"({SEARCH.lmr_full_researches}/{SEARCH.lmr_reductions})",
            file=sys.stderr,
        )

    return best_move, best_score, completed_depth, pv


def _format_score(score):
    if score > MATE_BOUND:
        return f"mate {(MATE_SCORE - score + 1) // 2}"
    if score < -MATE_BOUND:
        return f"mate -{(MATE_SCORE + score + 1) // 2}"
    return f"cp {score}"


def _extract_pv(board, root_move, limit):
    """Recover the principal variation for logging.

    The root move is passed in rather than looked up: the root loop searches
    ply 0 itself, so negamax never stores an entry for the root position.
    """
    line: list[str] = []
    scratch = board.copy(stack=False)
    if root_move is None or root_move not in scratch.legal_moves:
        return ""
    line.append(scratch.san(root_move))
    scratch.push(root_move)

    seen = set()
    for _ in range(limit - 1):
        key = chess.polyglot.zobrist_hash(scratch)
        if key in seen:
            break
        seen.add(key)
        index = key & TT_MASK
        if TT_DEPTH[index] < 0 or int(TT_KEY[index]) != key:
            break
        packed_move = int(TT_MOVE[index])
        if packed_move == 0:
            break
        move = unpack_move(packed_move)
        if move not in scratch.legal_moves:
            break
        line.append(scratch.san(move))
        scratch.push(move)
    return " ".join(line)


# =============================================================================
# OPPONENT TRACKING
# =============================================================================


class OpponentTracker:
    """Reconstructs what the opponent played between our turns.

    We are only ever handed a FEN, never a move list. But the process survives
    between our own moves, so we can keep the position we left behind, push
    every legal reply, and see which one produces the position we were just
    handed. That recovers the opponent's move, and by extension their whole
    game. Their think time is wall clock between our calls, which is close
    enough to be useful and is not exact.
    """

    def __init__(self):
        self.position_after_us = None  # board as we left it, opponent to move
        self.our_moves = []
        self.their_moves = []  # SAN
        self.their_move_times = []  # seconds, approximate
        self.last_reply_at = None
        self.fallbacks = 0

    def observe(self, board):
        """Return the opponent's last move, or None if unrecoverable."""
        if self.position_after_us is None:
            return None

        target = (board.board_fen(), board.turn, board.castling_rights)
        scratch = self.position_after_us

        for move in scratch.legal_moves:
            scratch.push(move)
            found = (scratch.board_fen(), scratch.turn,
                     scratch.castling_rights) == target
            scratch.pop()
            if found:
                self.their_moves.append(scratch.san(move))
                if self.last_reply_at is not None:
                    self.their_move_times.append(
                        time.perf_counter() - self.last_reply_at)
                return move
        return None

    def record_our_move(self, board, move):
        after = board.copy(stack=False)
        after.push(move)
        self.position_after_us = after
        self.our_moves.append(board.san(move))
        self.last_reply_at = time.perf_counter()


TRACKER = OpponentTracker()
GAME_KEYS: dict[int, int] = {}  # zobrist key -> occurrences this game

# =============================================================================
# TIME MANAGEMENT
# =============================================================================


def compute_budget(board, time_left_ms):
    """Return (soft_seconds, hard_seconds).

    Budgeted from the clock we were handed, not from a constant. The 5x5 engine
    used fixed SAFE_SOFT / SAFE_HARD values because its judge always handed it
    the same slice; here the clock actually varies, so the split is derived.
    """
    move_number = board.fullmove_number
    if move_number < 16:
        expected_left = 34
    elif move_number < 40:
        expected_left = 26
    else:
        expected_left = 20

    soft_ms = time_left_ms / expected_left + INCREMENT_MS * 0.7
    soft_ms = min(soft_ms, time_left_ms * 0.20)
    hard_ms = min(soft_ms * 3.0, time_left_ms * 0.33)

    # Time trouble overrides everything above. Round 49 reached move 74 with
    # 3.6s left, still handing itself 845ms of a 2560ms clock.
    if time_left_ms < LOW_CLOCK_MS:
        cap = INCREMENT_MS * 0.8 + time_left_ms * 0.02
        soft_ms = min(soft_ms, cap)
        hard_ms = min(hard_ms, cap * 1.6)

    ceiling = max(20.0, time_left_ms - SAFETY_MARGIN_MS)
    hard_ms = max(20.0, min(hard_ms, ceiling))
    soft_ms = max(10.0, min(soft_ms, hard_ms * 0.7))

    return soft_ms / 1000.0, hard_ms / 1000.0


# =============================================================================
# ENTRY POINT
# =============================================================================


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
    call_start = time.perf_counter()
    board = chess.Board(fen)

    legal = list(board.legal_moves)
    if not legal:
        # Should never happen; the referee would not ask. Fail legally rather
        # than crashing, since a crash is a loss.
        return "0000"

    fallback = legal[0].uci()

    try:
        their_move = TRACKER.observe(board)

        here = chess.polyglot.zobrist_hash(board)
        GAME_KEYS[here] = GAME_KEYS.get(here, 0) + 1
        SEARCH.game_keys = GAME_KEYS

        soft_s, hard_s = compute_budget(board, time_left_ms)

        if DEBUG:
            print(
                f"[move {board.fullmove_number} "
                f"{'w' if board.turn else 'b'}] clock {time_left_ms}ms "
                f"budget {soft_s * 1000:.0f}/{hard_s * 1000:.0f}ms "
                f"phase {game_phase(board)}/{MAX_PHASE} "
                f"legal {len(legal)}",
                file=sys.stderr,
            )
            if their_move is not None:
                took = (TRACKER.their_move_times[-1]
                        if TRACKER.their_move_times else 0.0)
                print(
                    f"  opponent played {TRACKER.their_moves[-1]} "
                    f"({their_move.uci()}) after ~{took:.2f}s",
                    file=sys.stderr,
                )
            elif TRACKER.position_after_us is not None:
                print("  opponent move could not be reconstructed",
                      file=sys.stderr)

        move, score, depth, _pv = search_root(board, soft_s, hard_s)

        if move is None or move not in board.legal_moves:
            move = legal[0]
            TRACKER.fallbacks += 1
            if DEBUG:
                print("  search returned no usable move; playing fallback",
                      file=sys.stderr)

        san = board.san(move)
        TRACKER.record_our_move(board, move)

        after = board.copy(stack=False)
        after.push(move)
        after_key = chess.polyglot.zobrist_hash(after)
        GAME_KEYS[after_key] = GAME_KEYS.get(after_key, 0) + 1

        if DEBUG:
            elapsed = (time.perf_counter() - call_start) * 1000
            print(
                f"  -> {san} ({move.uci()}) depth {depth} "
                f"{_format_score(score)} nodes {SEARCH.nodes} "
                f"(q {SEARCH.qnodes}) used {elapsed:.0f}ms of {time_left_ms}ms",
                file=sys.stderr,
            )

        return str(move.uci())

    except Exception as exc:  # a crash is a loss, so never let one escape
        if DEBUG:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"  recovered from {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        return fallback


# =============================================================================
# WARM UP
# =============================================================================
# Runs at import, inside the 90 second init budget, so the first real get_move
# is not paying for cold code paths and freshly grown internal caches.


def _warm_up():
    if DEBUG:
        # Always the first line of a game's log, on purpose - see BUILD_ID's
        # own comment.
        print(f"build {BUILD_ID}", file=sys.stderr)
    board = chess.Board()
    SEARCH.hard_deadline = time.perf_counter() + 3.0
    SEARCH.soft_deadline = SEARCH.hard_deadline
    SEARCH.game_keys = {}
    with contextlib.suppress(TimeUp):
        negamax(board, 4, -INFINITY, INFINITY, 0, None)
    tt_clear()
    SEARCH.path = []
    SEARCH.nodes = 0
    for ply in range(MAX_PLY):
        KILLERS[ply][0] = 0
        KILLERS[ply][1] = 0


_warm_up()


# =============================================================================
# MANUAL SELF TEST
# =============================================================================
# The platform imports this file, so __main__ never runs there. Locally:
#     uv run python agent.py                       start position, 5s
#     uv run python agent.py "<fen>" 3000          a position, 3s

if __name__ == "__main__":
    argv = sys.argv[1:]
    start_fen = argv[0] if argv else chess.STARTING_FEN
    budget_ms = int(argv[1]) if len(argv) > 1 else 5000

    probe = chess.Board(start_fen)
    # Plain ASCII, not board.unicode(): a Windows console is cp1252 and will
    # raise UnicodeEncodeError on the figurine characters.
    print(probe)
    print(f"\nfen   {probe.fen()}")
    print(f"eval  {evaluate(probe):+d} cp (side to move)")
    print(f"legal {probe.legal_moves.count()} moves")
    print(f"searching with {budget_ms}ms on the clock\n")

    chosen = get_move(start_fen, budget_ms)
    print(f"\nchose {chosen}  ({probe.san(chess.Move.from_uci(chosen))})")
