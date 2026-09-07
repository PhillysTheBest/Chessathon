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
import math
import sys
import time

import chess
import chess.polyglot

# =============================================================================
# TUNABLES
# =============================================================================

DEBUG = True  # writes to stderr; the platform keeps it in the per-game log

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

# Aspiration windows, DISABLED. The idea is sound and the node saving was real,
# 289804 -> 244629 nodes to reach depth 8, but it lost the arena that matters:
# +12 =10 -18 against philbert-v14 over 40 games, 42.5%, and one of those games
# was lost by flagging.
#
# The flag says the implementation is wrong rather than the idea. A fail low
# re-searches with a wider window, and nothing checks the clock between those
# re-searches, so a single ply can consume the rest of the budget. Worse, the
# move that comes back from a fail low is only an upper bound and search_root
# accepts it anyway.
#
# Setting the minimum depth above MAX_DEPTH_CAP disables it while leaving the
# code and the measurement in place. Re-enable only after the re-search loop
# respects the deadline and refuses to take a move from a failed window.
ASPIRATION_DELTA = 25
ASPIRATION_MIN_DEPTH = 999

TT_SIZE = 1 << 20
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
KILLERS = [[None, None] for _ in range(MAX_PLY)]
COUNTER = [[None] * 64 for _ in range(64)]

# Late move reduction table, indexed [depth][move_index].
LMR = [[0] * 64 for _ in range(64)]
for _d in range(1, 64):
    for _i in range(1, 64):
        LMR[_d][_i] = int(0.75 + math.log(_d) * math.log(_i) / 2.25)

# =============================================================================
# TRANSPOSITION TABLE
# =============================================================================
# Entries are (key, depth, score, flag, move), bucketed by key & TT_MASK and
# replaced on depth. Same scheme as the 5x5 engine, sized for 8x8.

# (key, depth, score, flag, move) per bucket, or None when empty.
TT: list[tuple[int, int, int, int, "chess.Move"] | None] = [None] * TT_SIZE


def tt_clear():
    for i in range(TT_SIZE):
        TT[i] = None


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

    def check_time(self):
        # Only touch the clock every 1024 nodes; perf_counter is not free.
        if self.nodes & 1023 == 0 and time.perf_counter() >= self.hard_deadline:
            raise TimeUp


SEARCH = Searcher()

# =============================================================================
# EVALUATION
# =============================================================================


def evaluate(board, alpha=-INFINITY, beta=INFINITY):
    """Static evaluation in centipawns, from the side to move's point of view.

    Two passes. The first is material, piece-square tables and the bishop pair,
    which is cheap and accounts for nearly all of the score in nearly every
    position. If that alone already sits a full LAZY_MARGIN outside the window
    the caller cares about, no amount of positional detail will drag it back
    in, so the second pass is skipped. Quiescence makes almost every evaluate
    call in the search and cuts on stand-pat most of the time, which is exactly
    the case this is for.

    Passing no window forces the full evaluation.
    """
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_pawns = board.pawns & white
    black_pawns = board.pawns & black

    mg = 0
    eg = 0
    phase = 0

    # ---- pass 1: material, PST, bishop pair -------------------------------
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color else -1
        pst_mg = PST_MG[color]
        pst_eg = PST_EG[color]

        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING):
            bb = board.pieces_mask(piece_type, color)
            if not bb:
                continue

            count = bb.bit_count()
            phase += PHASE_WEIGHT[piece_type] * count
            mg += sign * PIECE_VALUE_MG[piece_type] * count
            eg += sign * PIECE_VALUE_EG[piece_type] * count

            if piece_type == chess.BISHOP and count >= 2:
                mg += sign * BISHOP_PAIR_MG
                eg += sign * BISHOP_PAIR_EG

            table_mg = pst_mg[piece_type]
            table_eg = pst_eg[piece_type]
            sum_mg = 0
            sum_eg = 0
            for square in chess.scan_forward(bb):
                sum_mg += table_mg[square]
                sum_eg += table_eg[square]
            mg += sign * sum_mg
            eg += sign * sum_eg

    if phase > MAX_PHASE:
        phase = MAX_PHASE

    white_to_move = board.turn == chess.WHITE
    lazy = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    lazy += TEMPO if white_to_move else -TEMPO
    if not white_to_move:
        lazy = -lazy

    if lazy - LAZY_MARGIN >= beta or lazy + LAZY_MARGIN <= alpha:
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
        # The zone around the king we are attacking.
        target_zone = zone_of[color]

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

        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            weight_mg = MOBILITY_MG[piece_type]
            weight_eg = MOBILITY_EG[piece_type]
            attack_weight = KING_ATTACK_WEIGHT[piece_type]
            for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                attacked = board.attacks_mask(square)
                moves = (attacked & ~own).bit_count()
                mg += sign * weight_mg * moves
                eg += sign * weight_eg * moves
                # Same mask, reused: how much of the enemy king's zone this
                # piece covers. Nearly free, since attacks_mask is already paid
                # for by the mobility term.
                if target_zone:
                    hits = (attacked & target_zone).bit_count()
                    if hits:
                        attack_units[color] += attack_weight * hits

    # King danger is a midgame term only: with the queens off, a king in the
    # open is an asset, not a liability, and the endgame PSTs already say so.
    mg += (KING_DANGER[min(attack_units[chess.WHITE] // 8, 127)]
           - KING_DANGER[min(attack_units[chess.BLACK] // 8, 127)])

    # Taper between the midgame and endgame scores.
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    score += TEMPO if white_to_move else -TEMPO

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


def order_moves(board, moves, tt_move, ply, prev_move):
    """TT move, then captures by MVV-LVA, then promotions, then killers, then
    the counter move, then history. Alpha-beta only pays when this is right."""
    them = board.occupied_co[not board.turn]
    killer0, killer1 = KILLERS[ply] if ply < MAX_PLY else (None, None)
    counter = None
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
        elif move == killer0:
            score = 700_000_000
        elif move == killer1:
            score = 690_000_000
        elif counter is not None and move == counter:
            score = 680_000_000
        else:
            piece_type = board.piece_type_at(move.from_square)
            score = HISTORY[piece_type][move.from_square][to_square]

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


def _least_valuable(board, attackers, color):
    """Cheapest attacker of `color` in `attackers`, as (square, piece_type)."""
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING):
        subset = attackers & board.pieces_mask(piece_type, color)
        if subset:
            return subset.bit_length() - 1, piece_type
    return None, None


def see(board, move):
    """Material won by `move` in centipawns, assuming best play by both sides.

    Negative means the exchange loses material. Promotions and en passant are
    approximated: the point is ordering and pruning, not exactness.

    Pins are ignored, because attackers_mask does not know about them, so a
    defender that cannot legally recapture is still counted. Checked against a
    brute-force minimax of the exchange over 1083 captures from 300 random
    positions: 2 disagreements, both pins. That is the usual trade and the
    usual error rate; resolving pins here would cost more than it saves.
    """
    target = move.to_square
    from_square = move.from_square

    captured = board.piece_type_at(target)
    if captured is None:
        captured = chess.PAWN if board.is_en_passant(move) else 0

    attacker = board.piece_type_at(from_square)
    if attacker is None:
        return 0

    # gain[i] is the material balance after i captures, from the mover's view.
    gain = [SEE_VALUE[captured] if captured else 0]

    occupied = board.occupied & ~chess.BB_SQUARES[from_square]
    if board.is_en_passant(move):
        captured_square = target + (-8 if board.turn == chess.WHITE else 8)
        occupied &= ~chess.BB_SQUARES[captured_square]

    # Every piece bearing on the square, recomputed as the square is vacated so
    # a rook behind a rook joins the exchange when the front one takes.
    attackers = board.attackers_mask(chess.WHITE, target, occupied) | \
        board.attackers_mask(chess.BLACK, target, occupied)
    attackers &= occupied

    side = not board.turn
    on_square = attacker
    depth = 0

    while True:
        square, piece_type = _least_valuable(
            board, attackers & board.occupied_co[side], side)
        if square is None:
            break

        depth += 1
        # Standing on the square is worth what we just took, minus whatever the
        # opponent can win back by continuing.
        gain.append(SEE_VALUE[on_square] - gain[depth - 1])

        # No early exit here. The classic SEE pruning test is unsound the way
        # this loop is folded back: on a battery it stops one capture short and
        # reported Nxe5 in 1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 as
        # -65 when the true value is -225. Exchanges are short; play them out.
        on_square = piece_type
        occupied &= ~chess.BB_SQUARES[square]
        attackers = (board.attackers_mask(chess.WHITE, target, occupied)
                     | board.attackers_mask(chess.BLACK, target, occupied))
        attackers &= occupied
        side = not side

    # Fold back: at each step the side to move may decline the capture.
    while depth:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1

    return gain[0]


# =============================================================================
# QUIESCENCE
# =============================================================================

DELTA_MARGIN = 200


def qsearch(board, alpha, beta, ply):
    SEARCH.nodes += 1
    SEARCH.qnodes += 1
    SEARCH.check_time()

    if ply > SEARCH.seldepth:
        SEARCH.seldepth = ply
    if ply >= MAX_PLY - 1:
        return evaluate(board)

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
        stand_pat = evaluate(board, alpha, beta)
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

            board.push(move)
            score = -qsearch(board, -beta, -alpha, ply + 1)
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


def negamax(board, depth, alpha, beta, ply, prev_move, allow_null=True):
    SEARCH.nodes += 1
    SEARCH.check_time()

    is_pv = beta - alpha > 1
    original_alpha = alpha

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
        return evaluate(board)

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
    entry = TT[index]
    tt_move = None

    if entry is not None and entry[0] == key:
        tt_move = entry[4]
        if not is_pv and entry[1] >= depth and ply > 0:
            score = entry[2]
            # Mate scores are stored relative to the root; re-anchor them.
            if score > MATE_BOUND:
                score -= ply
            elif score < -MATE_BOUND:
                score += ply
            flag = entry[3]
            if flag == TT_EXACT:
                return score
            if flag == TT_LOWER and score >= beta:
                return score
            if flag == TT_UPPER and score <= alpha:
                return score

    if depth <= 0:
        return qsearch(board, alpha, beta, ply)

    static_eval = -MATE_SCORE if in_check else evaluate(board)

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
                board.push(chess.Move.null())
                SEARCH.path.append(key)
                try:
                    score = -negamax(board, depth - reduction - 1, -beta,
                                     -beta + 1, ply + 1, None, allow_null=False)
                finally:
                    SEARCH.path.pop()
                    board.pop()
                if score >= beta:
                    return beta if score > MATE_BOUND else score

    # --- internal iterative deepening -------------------------------------
    if tt_move is None and depth >= 4 and is_pv:
        negamax(board, depth - 2, alpha, beta, ply, prev_move, allow_null)
        entry = TT[index]
        if entry is not None and entry[0] == key:
            tt_move = entry[4]

    moves = list(board.legal_moves)
    if not moves:
        # Checkmate, or stalemate. Stalemate was a LOSS in the 5x5 variant;
        # in standard chess it is a draw and must score 0.
        return -MATE_SCORE + ply if in_check else 0

    moves = order_moves(board, moves, tt_move, ply, prev_move)

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
    best_move = moves[0]
    searched = 0

    SEARCH.path.append(key)
    try:
        for move_index, move in enumerate(moves):
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

            board.push(move)
            gives_check = board.is_check()

            if searched == 0:
                score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, move)
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
                                 -alpha, ply + 1, move)

                # The reduced search beat alpha, so redo it at full depth.
                if reduction > 0 and score > alpha:
                    score = -negamax(board, depth - 1, -alpha - 1, -alpha,
                                     ply + 1, move)

                # PVS re-search on a full window inside the bounds.
                if alpha < score < beta:
                    score = -negamax(board, depth - 1, -beta, -alpha,
                                     ply + 1, move)

            piece_type = board.piece_type_at(move.to_square)
            board.pop()
            searched += 1

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if is_quiet:
                            # Killer, counter move and history updates, lifted
                            # from the 5x5 engine's ordering scheme.
                            if ply < MAX_PLY:
                                slot = KILLERS[ply]
                                if slot[0] != move:
                                    slot[1] = slot[0]
                                    slot[0] = move
                            if prev_move is not None:
                                COUNTER[prev_move.from_square][
                                    prev_move.to_square] = move
                            if piece_type:
                                HISTORY[piece_type][move.from_square][
                                    move.to_square] += depth * depth
                        break
    finally:
        SEARCH.path.pop()

    if best_score == -INFINITY:
        # Everything was pruned. Return a static score rather than handing
        # -INFINITY back to the parent.
        return static_eval

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

    if entry is None or entry[0] != key or depth >= entry[1]:
        TT[index] = (key, depth, stored, flag, best_move)

    return best_score


# =============================================================================
# ROOT SEARCH
# =============================================================================


def _root_window(board, root_moves, best_move, depth, alpha, beta, base_stack):
    """Search every root move once inside (alpha, beta).

    Returns (move, score, finished, failed_high). A score at or below alpha is
    a fail low and a score at or above beta a fail high; either means the window
    was wrong and the caller has to widen it and come back.
    """
    iter_move = None
    iter_score = -INFINITY
    ordered = order_moves(board, root_moves, best_move, 0, None)

    for index, move in enumerate(ordered):
        board.push(move)
        try:
            if index == 0:
                score = -negamax(board, depth - 1, -beta, -alpha, 1, move)
            else:
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, 1, move)
                if alpha < score < beta:
                    score = -negamax(board, depth - 1, -beta, -alpha, 1, move)
        except TimeUp:
            return iter_move, iter_score, False, False
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


def search_root(board, soft_s, hard_s):
    """Iterative deepening. Returns (move, score, depth, pv)."""
    start = time.perf_counter()
    SEARCH.hard_deadline = start + hard_s
    SEARCH.soft_deadline = start + soft_s
    SEARCH.nodes = 0
    SEARCH.qnodes = 0
    SEARCH.seldepth = 0
    SEARCH.path = []

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

        while True:
            iter_move, iter_score, finished, failed_high = _root_window(
                board, root_moves, best_move, depth, alpha, beta, base_stack)
            if not finished or delta == 0:
                break
            if failed_high:
                delta *= 2
                beta = min(INFINITY, iter_score + delta)
                continue
            if iter_score <= alpha:
                # Fail low: the position is worse than assumed. What comes back
                # is an upper bound, so widen downwards and search again.
                delta *= 2
                alpha = max(-INFINITY, iter_score - delta)
                continue
            break

        # A partial iteration is still usable: the previous best was searched
        # first, so anything that beat it beat it on a real search.
        if iter_move is not None:
            best_move = iter_move
            best_score = iter_score

        if not finished:
            break

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
        entry = TT[key & TT_MASK]
        if entry is None or entry[0] != key or entry[4] is None:
            break
        move = entry[4]
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
        KILLERS[ply][0] = None
        KILLERS[ply][1] = None


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
