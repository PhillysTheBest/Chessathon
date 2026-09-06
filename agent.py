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
KING_SHIELD_MG = 9
TEMPO = 12

MOBILITY_MG = [0, 0, 4, 4, 2, 1, 0]
MOBILITY_EG = [0, 0, 4, 3, 4, 2, 0]

# Every square ahead of `square` on its own file and the two adjacent files.
PASSED_MASK = [[0] * 64 for _ in range(2)]
# Every square strictly ahead on the same file, for doubled pawns.
FRONT_FILE = [[0] * 64 for _ in range(2)]
# The three squares in front of a king, for the pawn shield.
KING_SHIELD = [[0] * 64 for _ in range(2)]
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
        self.game_keys = {}  # zobrist keys already played in the real game

    def check_time(self):
        # Only touch the clock every 1024 nodes; perf_counter is not free.
        if self.nodes & 1023 == 0 and time.perf_counter() >= self.hard_deadline:
            raise TimeUp


SEARCH = Searcher()

# =============================================================================
# EVALUATION
# =============================================================================


def evaluate(board):
    """Static evaluation in centipawns, from the side to move's point of view."""
    mg = 0
    eg = 0
    phase = 0

    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color else -1
        own = board.occupied_co[color]
        own_pawns = white_pawns if color else black_pawns
        their_pawns = black_pawns if color else white_pawns

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

            table_mg = pst_mg[piece_type]
            table_eg = pst_eg[piece_type]
            mobile = piece_type in (chess.KNIGHT, chess.BISHOP,
                                    chess.ROOK, chess.QUEEN)

            for square in chess.scan_forward(bb):
                mg += sign * table_mg[square]
                eg += sign * table_eg[square]

                if piece_type == chess.PAWN:
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

                elif piece_type == chess.ROOK:
                    file_bb = chess.BB_FILES[chess.square_file(square)]
                    if not (file_bb & own_pawns):
                        if file_bb & their_pawns:
                            mg += sign * ROOK_SEMI_MG
                            eg += sign * ROOK_SEMI_EG
                        else:
                            mg += sign * ROOK_OPEN_MG
                            eg += sign * ROOK_OPEN_EG

                elif piece_type == chess.KING:
                    shield = (KING_SHIELD[color][square] & own_pawns).bit_count()
                    mg += sign * KING_SHIELD_MG * shield

                if mobile:
                    moves = (board.attacks_mask(square) & ~own).bit_count()
                    mg += sign * MOBILITY_MG[piece_type] * moves
                    eg += sign * MOBILITY_EG[piece_type] * moves

        if board.pieces_mask(chess.BISHOP, color).bit_count() >= 2:
            mg += sign * BISHOP_PAIR_MG
            eg += sign * BISHOP_PAIR_EG

    # Taper between the midgame and endgame scores.
    if phase > MAX_PHASE:
        phase = MAX_PHASE
    score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    score += TEMPO if board.turn == chess.WHITE else -TEMPO

    return score if board.turn == chess.WHITE else -score


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

    if in_check:
        # Never stand pat in check. Search every evasion so we do not call a
        # position quiet while the king is attacked.
        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply
        moves = order_moves(board, moves, None, ply, None)
        best = -INFINITY
    else:
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        best = stand_pat
        moves = order_captures(board, list(board.generate_legal_captures()))

    for move in moves:
        if not in_check:
            # Delta pruning: if winning this piece outright still leaves us far
            # below alpha, the capture cannot rescue the position.
            victim = board.piece_type_at(move.to_square)
            gain = PIECE_VALUE_MG[victim] if victim else PIECE_VALUE_MG[chess.PAWN]
            if move.promotion:
                gain += PIECE_VALUE_MG[move.promotion]
            if best + gain + DELTA_MARGIN < alpha:
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
        # A repeat inside the search path, or of a position already played in
        # the real game, is a draw. The referee claims threefold automatically,
        # so this is also what stops us shuffling a won game away.
        if key in SEARCH.path or key in SEARCH.game_keys:
            return 0

    if ply >= MAX_PLY - 1:
        return evaluate(board)

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

    in_check = board.is_check()

    # Check extension, capped the same way the 5x5 engine capped it.
    if in_check and ply < 40:
        depth += 1

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


def search_root(board, soft_s, hard_s):
    """Iterative deepening. Returns (move, score, depth, pv)."""
    start = time.perf_counter()
    SEARCH.hard_deadline = start + hard_s
    SEARCH.soft_deadline = start + soft_s
    SEARCH.nodes = 0
    SEARCH.qnodes = 0
    SEARCH.seldepth = 0
    SEARCH.path = []

    moves = list(board.legal_moves)
    if not moves:
        return None, 0, 0, ""

    best_move = moves[0]
    best_score = 0
    completed_depth = 0
    pv = ""

    for depth in range(1, MAX_DEPTH_CAP + 1):
        try:
            score = negamax(board, depth, -INFINITY, INFINITY, 0, None)
        except TimeUp:
            break

        key = chess.polyglot.zobrist_hash(board)
        entry = TT[key & TT_MASK]
        if (entry is not None and entry[0] == key and entry[4] is not None
                and entry[4] in board.legal_moves):
            best_move = entry[4]
        best_score = score
        completed_depth = depth
        pv = _extract_pv(board, 8)

        if DEBUG:
            elapsed = time.perf_counter() - start
            nps = int(SEARCH.nodes / elapsed) if elapsed > 0 else 0
            print(
                f"  depth {depth:2d} seldepth {SEARCH.seldepth:2d} "
                f"score {_format_score(score)} nodes {SEARCH.nodes} "
                f"nps {nps} time {elapsed * 1000:.0f}ms pv {pv}",
                file=sys.stderr,
            )

        # A forced mate is found; no deeper search can improve on it.
        if abs(score) > MATE_BOUND:
            break

        # Soft limit: do not start an iteration we almost certainly cannot
        # finish. Each extra ply costs roughly 2-3x the last one.
        if time.perf_counter() - start > soft_s * 0.5:
            break

    return best_move, best_score, completed_depth, pv


def _format_score(score):
    if score > MATE_BOUND:
        return f"mate {(MATE_SCORE - score + 1) // 2}"
    if score < -MATE_BOUND:
        return f"mate -{(MATE_SCORE + score + 1) // 2}"
    return f"cp {score}"


def _extract_pv(board, limit):
    """Walk the transposition table to recover the principal variation."""
    line: list[str] = []
    scratch = board.copy(stack=False)
    seen = set()
    for _ in range(limit):
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
GAME_KEYS: dict[int, bool] = {}

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

        GAME_KEYS[chess.polyglot.zobrist_hash(board)] = True
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
        GAME_KEYS[chess.polyglot.zobrist_hash(after)] = True

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
