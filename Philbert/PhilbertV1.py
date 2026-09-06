import random
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from extension.board_utils import take_notes, list_legal_moves_for

# =============================================================================
# CONSTANTS & BASIC SETUP
# =============================================================================

ColorIndex = int  # 0 = white, 1 = black
PieceIndex = int  # 0..5 = P,N,B,R,Q,K
Square = int      # 0..24
Bitboard = int
Move = int

WHITE, BLACK = 0, 1
PIECE_P, PIECE_N, PIECE_B, PIECE_R, PIECE_Q, PIECE_K = range(6)



W, H = 5, 5
SQUARES = W * H
MATE_SCORE = 99999

# Depth / search caps
MAX_DEPTH_CAP = 10
MAX_PLY = 256

# Avoid timeouts
SOFT_TIME_LIMIT = 11.5
HARD_TIME_LIMIT = 12.5

# =============================================================================
# REQUIRED GLOBAL CONSTANTS (TT, SAFE TIME, PIECE MAP)
# =============================================================================

# --- Transposition table size ---
TT_SIZE = 1 << 18          # 262,144 entries
TT_MASK = TT_SIZE - 1

# --- Safe time limits for the judge ---
SAFE_SOFT = 10.5           # stop deepening after this
SAFE_HARD = 11.5           # absolutely must return before 12.5s kill

# --- Piece char → internal piece index ---
PIECE_FROM_CHAR = {
    "P": PIECE_P,
    "N": PIECE_N,
    "B": PIECE_B,
    "R": PIECE_R,   # right piece = rook+knight
    "Q": PIECE_Q,
    "K": PIECE_K,
}

# ============================================================
# File masks (5 files for 5×5 board)
# ============================================================

FILE_MASKS = [
    sum(1 << (5*y + 0) for y in range(5)),  # File A (x=0)
    sum(1 << (5*y + 1) for y in range(5)),  # File B (x=1)
    sum(1 << (5*y + 2) for y in range(5)),  # File C (x=2)
    sum(1 << (5*y + 3) for y in range(5)),  # File D (x=3)
    sum(1 << (5*y + 4) for y in range(5)),  # File E (x=4)
]

RANK_MASKS = [
    sum(1 << (5*0 + x) for x in range(5)),  # Rank 0
    sum(1 << (5*1 + x) for x in range(5)),  # Rank 1
    sum(1 << (5*2 + x) for x in range(5)),  # Rank 2
    sum(1 << (5*3 + x) for x in range(5)),  # Rank 3
    sum(1 << (5*4 + x) for x in range(5)),  # Rank 4
]


# =============================================================================
# =============================================================================

# HISTORY[piece][from][to]
HISTORY = [
    [
        [0 for _ in range(SQUARES)]
        for _ in range(SQUARES)
    ]
    for _ in range(6)
]

# CONT_HISTORY[piece_prev][from_prev][to_prev][to_current]
CONT_HISTORY = [
    [
        [
            [0 for _ in range(SQUARES)]
            for _ in range(SQUARES)
        ]
        for _ in range(SQUARES)
    ]
    for _ in range(6)
]

# =============================================================================
# BASIC GEOMETRY
# =============================================================================

def in_board(x: int, y: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < W and 0 <= y < H

def sq_to_xy(sq: Square) -> Tuple[int, int]:
    """Convert square index to (x, y) coordinates."""
    return sq % W, sq // W

def xy_to_sq(x: int, y: int) -> Square:
    """Convert (x, y) coordinates to square index."""
    return y * W + x

# Knight and King offsets
KNIGHT_OFFS = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
KING_OFFS   = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

PRE_KNIGHT = [[] for _ in range(SQUARES)]
PRE_KING   = [[] for _ in range(SQUARES)]

for sq in range(SQUARES):
    x, y = sq_to_xy(sq)
    for dx, dy in KNIGHT_OFFS:
        nx, ny = x+dx, y+dy
        if in_board(nx, ny):
            PRE_KNIGHT[sq].append(xy_to_sq(nx, ny))
    for dx, dy in KING_OFFS:
        nx, ny = x+dx, y+dy
        if in_board(nx, ny):
            PRE_KING[sq].append(xy_to_sq(nx, ny))

# Precompute Attack Bitmasks (Perf Optimization)
KNIGHT_ATTACKS = [0] * SQUARES
KING_ATTACKS = [0] * SQUARES

for sq in range(SQUARES):
    # Knight
    m = 0
    for t in PRE_KNIGHT[sq]:
        m |= (1 << t)
    KNIGHT_ATTACKS[sq] = m
    
    # King
    m = 0
    for t in PRE_KING[sq]:
        m |= (1 << t)
    KING_ATTACKS[sq] = m

def attacks_from_knight(sq: int) -> int:
    """Return bitmask of knight attacks from square sq."""
    return KNIGHT_ATTACKS[sq]

def attacks_from_king(sq: int) -> int:
    """Return bitmask of king attacks from square sq."""
    return KING_ATTACKS[sq]

_rng = random.Random(93377)
PAWN_Z_W = [_rng.getrandbits(64) for _ in range(25)]
PAWN_Z_B = [_rng.getrandbits(64) for _ in range(25)]



# =============================================================================
# PST (MIDGAME + ENDGAME) + TAPERED EVAL
# =============================================================================

PHASE_WEIGHTS = [0, 2, 2, 4, 4, 0]   # pawn=0, knight=2, bishop=2, right=4, queen=4, king=0
MAX_PHASE = 24 # sum(PHASE_WEIGHTS) * 2

PIECE_VALUE_MG = [100, 360, 340, 850, 900, 0]
PIECE_VALUE_EG = [140, 340, 330, 830, 880, 0]

# White PSTs (5x5)
PAWN_PST_MG_W = [
    [0,0,0,0,0],
    [4,5,6,5,4],
    [2,3,4,3,2],
    [0,1,1,1,0],
    [0,0,0,0,0],
]
PAWN_PST_EG_W = [
    [0,0,0,0,0],
    [3,4,5,4,3],
    [2,3,4,3,2],
    [1,2,2,2,1],
    [0,0,0,0,0],
]
KNIGHT_PST_MG_W = [
    [-4,-2,-2,-2,-4],
    [-2, 1, 2, 1,-2],
    [-2, 2, 4, 2,-2],
    [-2, 1, 2, 1,-2],
    [-4,-2,-2,-2,-4],
]
KNIGHT_PST_EG_W = [
    [-3,-1,-1,-1,-3],
    [-1, 1, 2, 1,-1],
    [-1, 2, 3, 2,-1],
    [-1, 1, 2, 1,-1],
    [-3,-1,-1,-1,-3],
]
RIGHT_PST_MG_W = [
    [0,1,2,1,0],
    [1,3,4,3,1],
    [1,4,6,4,1],
    [1,3,4,3,1],
    [0,1,2,1,0],
]
RIGHT_PST_EG_W = [
    [0,1,2,1,0],
    [1,2,3,2,1],
    [1,3,4,3,1],
    [1,2,3,2,1],
    [0,1,2,1,0],
]
KING_PST_MG_W = [
    [0, 2, 2, 2, 0],
    [2, 0, 0, 0, 2],
    [2, -2, -15, -2, 2], # Center penalty increased (-4 -> -15)
    [2, 0, 0, 0, 2],
    [0, 2, 4, 2, 0],
]
KING_PST_EG_W = [
    [-2,-1,0,-1,-2],
    [-1, 1,1, 1,-1],
    [ 0, 1,2, 1,0],
    [-1, 1,1, 1,-1],
    [-2,-1,0,-1,-2],
]

PST_MG = [[[0]*SQUARES for _ in range(6)] for _ in range(2)]
PST_EG = [[[0]*SQUARES for _ in range(6)] for _ in range(2)]


def _init_psts() -> None:
    """Initialize Piece-Square Tables."""
    for sq in range(SQUARES):
        x, y = sq_to_xy(sq)
        # White pieces
        PST_MG[WHITE][PIECE_P][sq] = PAWN_PST_MG_W[y][x]
        PST_EG[WHITE][PIECE_P][sq] = PAWN_PST_EG_W[y][x]
        PST_MG[WHITE][PIECE_N][sq] = KNIGHT_PST_MG_W[y][x]
        PST_EG[WHITE][PIECE_N][sq] = KNIGHT_PST_EG_W[y][x]
        PST_MG[WHITE][PIECE_R][sq] = RIGHT_PST_MG_W[y][x]
        PST_EG[WHITE][PIECE_R][sq] = RIGHT_PST_EG_W[y][x]
        PST_MG[WHITE][PIECE_K][sq] = KING_PST_MG_W[y][x]
        PST_EG[WHITE][PIECE_K][sq] = KING_PST_EG_W[y][x]

        # Mirror vertically for black
        my = H - 1 - y
        msq = xy_to_sq(x, my)
        PST_MG[BLACK][PIECE_P][msq] = PAWN_PST_MG_W[y][x]
        PST_EG[BLACK][PIECE_P][msq] = PAWN_PST_EG_W[y][x]
        PST_MG[BLACK][PIECE_N][msq] = KNIGHT_PST_MG_W[y][x]
        PST_EG[BLACK][PIECE_N][msq] = KNIGHT_PST_EG_W[y][x]
        PST_MG[BLACK][PIECE_R][msq] = RIGHT_PST_MG_W[y][x]
        PST_EG[BLACK][PIECE_R][msq] = RIGHT_PST_EG_W[y][x]
        PST_MG[BLACK][PIECE_K][msq] = KING_PST_MG_W[y][x]
        PST_EG[BLACK][PIECE_K][msq] = KING_PST_EG_W[y][x]

_init_psts()

# ============================================================
# Pawn structure evaluation constants (5×5 tuned values)
# ============================================================

# Doubled pawns penalty
EVAL_DOUBLED_MG = -10
EVAL_DOUBLED_EG = -20
DOUBLED_PAWN = -10

# Isolated pawn penalty
EVAL_ISOLATED_MG = -10
EVAL_ISOLATED_EG = -20
ISOLATED_PAWN = -10

# Backward pawn penalty
EVAL_BACKWARD_MG = -10
EVAL_BACKWARD_EG = -20
BACKWARD_PAWN = -8

# Bonuses for passed pawns by rank (distance to promotion)
# Rank 0 = base rank, Rank 4 = promotion square
PASS_MG = [0, 10, 30, 60, 100]
PASS_EG = [0, 20, 60, 120, 200]

# Connected passed pawns bonus
CONNECTED_PASS_MG = 15
CONNECTED_PASS_EG = 30

# Protected passed pawn bonus
PROTECTED_PASS_MG = 20
PROTECTED_PASS_EG = 40


# =============================================================================
# BITBOARD UTILS
# =============================================================================

def bit_scan(bb: Bitboard) -> int:
    """Yield LS1B indices from bitboard."""
    while bb:
        lsb = bb & -bb
        sq = lsb.bit_length() - 1
        yield sq
        bb &= bb - 1

# =============================================================================
# SLIDING ATTACK TABLES (FAST PRECOMPUTED)
# =============================================================================

ROOK_RAYS = [[] for _ in range(SQUARES)]
BISHOP_RAYS = [[] for _ in range(SQUARES)]

for sq in range(SQUARES):
    x, y = sq_to_xy(sq)
    # Rook rays
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        cx, cy = x+dx, y+dy
        while in_board(cx, cy):
            ROOK_RAYS[sq].append(xy_to_sq(cx,cy))
            cx += dx
            cy += dy
    # Bishop rays
    for dx, dy in ((1,1),(1,-1),(-1,1),(-1,-1)):
        cx, cy = x+dx, y+dy
        while in_board(cx, cy):
            BISHOP_RAYS[sq].append(xy_to_sq(cx,cy))
            cx += dx
            cy += dy

# Attack tables stored as 2D arrays [sq][occ_mask]
ROOK_ATTACKS = []
BISHOP_ATTACKS = []

for sq in range(SQUARES):
    # --- ROOK ---
    rel = ROOK_RAYS[sq]
    n = len(rel)
    table = [0] * (1 << n)
    for mask in range(1 << n):
        occ = 0
        for i in range(n):
            if mask & (1 << i):
                occ |= (1 << rel[i])
        
        # Ray casting
        att = 0
        x, y = sq_to_xy(sq)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            cx, cy = x+dx, y+dy
            while in_board(cx, cy):
                s2 = xy_to_sq(cx, cy)
                att |= (1 << s2)
                if occ & (1 << s2):
                    break
                cx += dx
                cy += dy
        table[mask] = att
    ROOK_ATTACKS.append(table)

    # --- BISHOP ---
    rel = BISHOP_RAYS[sq]
    n = len(rel)
    table = [0] * (1 << n)
    for mask in range(1 << n):
        occ = 0
        for i in range(n):
            if mask & (1 << i):
                occ |= (1 << rel[i])
        
        # Ray casting
        att = 0
        x, y = sq_to_xy(sq)
        for dx, dy in ((1,1),(1,-1),(-1,1),(-1,-1)):
            cx, cy = x+dx, y+dy
            while in_board(cx, cy):
                s2 = xy_to_sq(cx, cy)
                att |= (1 << s2)
                if occ & (1 << s2):
                    break
                cx += dx
                cy += dy
        table[mask] = att
    BISHOP_ATTACKS.append(table)

# --- VERIFICATION ---
# Verify (4,2) Bishop attacks do NOT include (2,1)
# (4,2) is sq 14. (2,1) is sq 7.
_q_sq = 14
_bad_sq = 7
_rays = BISHOP_RAYS[_q_sq]
if _bad_sq in _rays:
    raise RuntimeError(f"FATAL: Bishop Rays for 14 include 7! Rays: {_rays}")
# Check attack table for empty board
_empty_att = BISHOP_ATTACKS[_q_sq][0]
if _empty_att & (1 << _bad_sq):
    raise RuntimeError(f"FATAL: Bishop Attacks for 14 include 7! Att: {bin(_empty_att)}")

# Verify (0,4) Bishop attacks INCLUDE (3,1)
# (0,4) is sq 20. (3,1) is sq 8.
_k_sq = 20
_q_sq_check = 8
_rays_check = BISHOP_RAYS[_k_sq]
if _q_sq_check not in _rays_check:
    raise RuntimeError(f"FATAL: Bishop Rays for 20 DO NOT include 8! Rays: {_rays_check}")
# Check attack table for empty board
_empty_att_check = BISHOP_ATTACKS[_k_sq][0]
if not (_empty_att_check & (1 << _q_sq_check)):
    raise RuntimeError(f"FATAL: Bishop Attacks for 20 DO NOT include 8! Att: {bin(_empty_att_check)}")



def rook_attacks_from(sq: int, occ_all: int) -> int:
    """
    Ultra-fast rook attacks using precomputed ray masks.
    Cython-friendly: tight loop over ints, no allocations.
    """
    rel = ROOK_RAYS[sq]   # list[int]
    idx = 0
    bit = 1

    for s in rel:
        if occ_all & (1 << s):
            idx |= bit
        bit <<= 1

    return ROOK_ATTACKS[sq][idx]


def bishop_attacks_from(sq: int, occ_all: int) -> int:
    """
    Ultra-fast bishop attacks using precomputed ray masks.
    Same pattern as rook_attacks_from.
    """
    rel = BISHOP_RAYS[sq]   # list[int]
    idx = 0
    bit = 1

    for s in rel:
        if occ_all & (1 << s):
            idx |= bit
        bit <<= 1

    return BISHOP_ATTACKS[sq][idx]





# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class UndoInfo:
    piece: int
    captured: int
    from_sq: int
    to_sq: int
    promoted: int
    prev_zkey: int
    prev_mg: int
    prev_eg: int
    prev_phase: int
    prev_occ_w: int
    prev_occ_b: int
    prev_occ: int
    prev_wtm: bool
    prev_mob_w: int
    prev_mob_b: int
    prev_wk_sq: int
    prev_bk_sq: int

class GameState:
    """
    Ultra-fast pure integer state.
    Replaces GameState completely.
    """
    __slots__ = (
        # White pieces
        'WP','WN','WB','WR','WQ','WK',
        # Black pieces
        'BP','BN','BB','BR','BQ','BK',
        'pw','pb',
        # Occupancy
        'occ_w','occ_b','occ_all',
        # Eval parts
        'mg','eg','phase',
        # Zobrist
        'zkey',
        'mob_w','mob_b',
        'history',
        'white_to_move',
        'wk_sq', 'bk_sq'
    )

    def __init__(self,
                 WP: Bitboard, WN: Bitboard, WB: Bitboard, WR: Bitboard, WQ: Bitboard, WK: Bitboard,
                 BP: Bitboard, BN: Bitboard, BB: Bitboard, BR: Bitboard, BQ: Bitboard, BK: Bitboard,
                 occ_w: int, occ_b: int, occ_all: int,
                 mg: int, eg: int, phase: int, zkey: int, white_to_move: bool):
        self.WP = WP
        self.WN = WN
        self.WB = WB
        self.WR = WR
        self.WQ = WQ
        self.WK = WK
        self.wk_sq = (WK & -WK).bit_length() - 1 if WK else -1

        self.BP = BP
        self.BN = BN
        self.BB = BB
        self.BR = BR
        self.BQ = BQ
        self.BK = BK
        self.bk_sq = (BK & -BK).bit_length() - 1 if BK else -1

        self.pw = [WP, WN, WB, WR, WQ, WK]
        self.pb = [BP, BN, BB, BR, BQ, BK]

        self.occ_w = occ_w
        self.occ_b = occ_b
        self.occ_all = occ_all

        self.mg = mg
        self.eg = eg
        self.phase = phase

        self.zkey = zkey
        self.mob_w = 0
        self.mob_b = 0
        self.history = [] # List of Zobrist keys for repetition detection
        self.white_to_move = white_to_move




random.seed(2025)

Z_PIECE = [[[0]*SQUARES for _ in range(6)] for _ in range(2)]
for c in (WHITE, BLACK):
    for p in range(6):
        for sq in range(SQUARES):
            Z_PIECE[c][p][sq] = random.getrandbits(64)

Z_SIDE = random.getrandbits(64)

def zflip_piece(state: GameState, color: int, piece: int, sq: int) -> None:
    """XOR Zobrist key with piece signature."""
    state.zkey ^= Z_PIECE[color][piece][sq]

def zflip_side(state: GameState) -> None:
    """XOR Zobrist key with side-to-move signature."""
    state.zkey ^= Z_SIDE

def zflip_remove(bb: GameState, color: int, piece: int, sq: int) -> None:
    """Update Zobrist key for removing a piece."""
    bb.zkey ^= Z_PIECE[color][piece][sq]

def zflip_place(bb: GameState, color: int, piece: int, sq: int) -> None:
    """Update Zobrist key for placing a piece."""
    bb.zkey ^= Z_PIECE[color][piece][sq]

def zflip_capture(bb: GameState, color_captured: int, piece: int, sq: int) -> None:
    """Update Zobrist key for capturing a piece."""
    # XOR out the captured piece
    bb.zkey ^= Z_PIECE[color_captured][piece][sq]
def zflip_promote(bb: GameState, color: int, fr: int, to: int) -> None:
    """Update Zobrist key for pawn promotion."""
    # pawn leaves fr
    bb.zkey ^= Z_PIECE[color][PIECE_P][fr]

    # queen arrives to
    bb.zkey ^= Z_PIECE[color][PIECE_Q][to]


def zflip_unpromote(bb: GameState, color: int, fr: int, to: int) -> None:
    """Update Zobrist key for undoing info promotion."""
    # queen leaves to
    bb.zkey ^= Z_PIECE[color][PIECE_Q][to]

    # pawn returns to fr
    bb.zkey ^= Z_PIECE[color][PIECE_P][fr]











# =============================================================================
# GameState (bitboard position)
# =============================================================================






def inc_eval_after_move(bb: GameState, color: int, piece: int, fr: int, to: int,
                        captured: int, promoted: bool) -> None:
    """Update mg, eg, phase incrementally."""
    sign = 1 if color == WHITE else -1
    enemy = BLACK if color == WHITE else WHITE

    # remove moving piece old square
    bb.mg -= sign * (PIECE_VALUE_MG[piece] + PST_MG[color][piece][fr])
    bb.eg -= sign * (PIECE_VALUE_EG[piece] + PST_EG[color][piece][fr])
    bb.phase -= PHASE_WEIGHTS[piece]

    # add new
    newp = PIECE_Q if promoted else piece
    bb.mg += sign * (PIECE_VALUE_MG[newp] + PST_MG[color][newp][to])
    bb.eg += sign * (PIECE_VALUE_EG[newp] + PST_EG[color][newp][to])
    bb.phase += PHASE_WEIGHTS[newp]

    # capture
    if captured != CAPTURE_NONE:
        esign = -sign
        bb.mg -= esign * (PIECE_VALUE_MG[captured] + PST_MG[enemy][captured][to])
        bb.eg -= esign * (PIECE_VALUE_EG[captured] + PST_EG[enemy][captured][to])
        bb.phase -= PHASE_WEIGHTS[captured]

    if bb.phase < 0: bb.phase = 0
    if bb.phase > MAX_PHASE: bb.phase = MAX_PHASE

    # incremental mobility update
    bb.mob_w = mobility_side(WHITE, bb)
    bb.mob_b = mobility_side(BLACK, bb)



def mobility_side(color: int, bb: GameState) -> int:
    """Calculate mobility score for a side."""
    mob = 0
    occ = bb.occ_all

    if color == WHITE:
        P = (bb.WP, bb.WN, bb.WB, bb.WR, bb.WQ, bb.WK)
        opp_occ = bb.occ_b
        own_occ = bb.occ_w
        dy = -1
    else:
        P = (bb.BP, bb.BN, bb.BB, bb.BR, bb.BQ, bb.BK)
        opp_occ = bb.occ_w
        own_occ = bb.occ_b
        dy = 1

    # ------------------------
    # Pawn mobility: quiet pushes only
    # ------------------------
    b = P[PIECE_P]
    while b:
        lsb = b & -b
        fr = lsb.bit_length() - 1
        b ^= lsb

        x = fr % 5
        y = fr // 5
        ny = y + dy
        if 0 <= ny < 5:
            to = ny * 5 + x
            if not (occ & (1 << to)):
                mob += 1

    # ------------------------
    # Knight mobility
    # ------------------------
    b = P[PIECE_N]
    while b:
        lsb = b & -b
        fr = lsb.bit_length() - 1
        b ^= lsb
        
        att = attacks_from_knight(fr) & ~own_occ
        mob += att.bit_count()

    # ------------------------
    # Bishop mobility
    # ------------------------
    b = P[PIECE_B]
    while b:
        lsb = b & -b
        fr = lsb.bit_length() - 1
        b ^= lsb
        
        att = bishop_attacks_from(fr, occ) & ~own_occ
        mob += att.bit_count()

    # ------------------------
    # Right Piece (Rook+Knight) mobility
    # ------------------------
    b = P[PIECE_R]
    while b:
        lsb = b & -b
        fr = lsb.bit_length() - 1
        b ^= lsb
        
        # Right piece moves like Rook + Knight
        att = (rook_attacks_from(fr, occ) | attacks_from_knight(fr)) & ~own_occ
        mob += att.bit_count()

    # ------------------------
    # Queen mobility
    # ------------------------
    b = P[PIECE_Q]
    while b:
        lsb = b & -b
        fr = lsb.bit_length() - 1
        b ^= lsb

        att = (rook_attacks_from(fr, occ) |
               bishop_attacks_from(fr, occ))
        att &= ~own_occ
        mob += att.bit_count()

    # ------------------------
    # King mobility
    # ------------------------
    b = P[PIECE_K]
    if b:
        fr = (b & -b).bit_length() - 1
        # Use precomputed king moves
        att = attacks_from_king(fr) & ~own_occ
        mob += att.bit_count()

    return mob





# =============================================================================
# MOVE PACKING (FAST INT-BASED)
# =============================================================================

MOVE_TO_SHIFT   = 0
MOVE_FROM_SHIFT = 5
MOVE_ISCAP_SHIFT = 10
MOVE_PIECE_SHIFT = 11
MOVE_COLOR_SHIFT = 14
MOVE_CAPT_SHIFT = 15
MOVE_PROMO_SHIFT = 18

MOVE_TO_MASK = 0x1F
MOVE_FROM_MASK = 0x1F
MOVE_PIECE_MASK = 0x7
MOVE_COLOR_MASK = 0x1
MOVE_CAPT_MASK = 0x7
MOVE_PROMO_MASK = 0x1

CAPTURE_NONE = 7

def encode_move(color: int, piece: int, fr: int, to: int, is_cap: int, captured: int, promoted: int = 0) -> Move:
    """Pack move data into a 32-bit integer."""
    return (
        (to  << MOVE_TO_SHIFT) |
        (fr  << MOVE_FROM_SHIFT) |
        (is_cap << MOVE_ISCAP_SHIFT) |
        (piece  << MOVE_PIECE_SHIFT) |
        (color  << MOVE_COLOR_SHIFT) |
        (captured << MOVE_CAPT_SHIFT) |
        (promoted << MOVE_PROMO_SHIFT)
    )

def move_to(m: Move) -> int:
    """Extract destination square from move."""
    return (m >> MOVE_TO_SHIFT) & MOVE_TO_MASK

def move_from(m: Move) -> int:
    """Extract origin square from move."""
    return (m >> MOVE_FROM_SHIFT) & MOVE_FROM_MASK

def move_is_capture(m: Move) -> bool:
    """Check if move is a capture."""
    return ((m >> MOVE_ISCAP_SHIFT) & 1) != 0

def move_is_promotion(m: Move) -> bool:
    """Check if move is a promotion."""
    return ((m >> MOVE_PROMO_SHIFT) & MOVE_PROMO_MASK) != 0

def move_piece(m: Move) -> int:
    """Extract piece type from move."""
    return (m >> MOVE_PIECE_SHIFT) & MOVE_PIECE_MASK

def move_color(m: Move) -> int:
    """Extract color from move."""
    return (m >> MOVE_COLOR_SHIFT) & MOVE_COLOR_MASK

def move_captured_piece(m: Move) -> int:
    """Extract captured piece type from move."""
    return (m >> MOVE_CAPT_SHIFT) & MOVE_CAPT_MASK

def move_promo(m: Move) -> int:
    """Extract promotion piece type from move."""
    return (m >> MOVE_PROMO_SHIFT) & MOVE_PROMO_MASK

# =============================================================================
# MOVE LOGIC & GENERATION
# =============================================================================

def attacks_from_pawn(color: int, sq: int) -> int:
    """Return bitmask of pawn attacks from square sq."""
    x, y = sq_to_xy(sq)
    if color == WHITE:
        ny = y - 1
        if ny < 0: return 0
        mask = 0
        if x > 0: mask |= (1 << xy_to_sq(x - 1, ny))
        if x < 4: mask |= (1 << xy_to_sq(x + 1, ny))
        return mask
    else:
        ny = y + 1
        if ny >= 5: return 0
        mask = 0
        if x > 0: mask |= (1 << xy_to_sq(x - 1, ny))
        if x < 4: mask |= (1 << xy_to_sq(x + 1, ny))
        return mask

def attacks_from_knight(sq: int) -> int:
    mask = 0
    for t in PRE_KNIGHT[sq]:
        mask |= (1 << t)
    return mask

def attacks_from_king(sq: int) -> int:
    mask = 0
    for t in PRE_KING[sq]:
        mask |= (1 << t)
    return mask

def in_check(bb: GameState, color: int) -> bool:
    """
    Is `color`'s king currently in check in position `bb`?
    Uses bitboard attacks only, no allocations.
    """
    enemy = BLACK if color == WHITE else WHITE

    # Locate king square from cache
    if color == WHITE:
        ksq = bb.wk_sq
    else:
        ksq = bb.bk_sq

    # If king missing (captured?), treat as "in check" (illegal state)
    if ksq == -1:
        return True
    occ_all = bb.occ_all

    # Pawn attacks
    if color == WHITE:
        pawns = bb.pb[PIECE_P]
    else:
        pawns = bb.pw[PIECE_P]

    if pawns and (attacks_from_pawn(color, ksq) & pawns):
        return True

    # Knight attacks (from Knights AND Right pieces)
    if color == WHITE:
        knights = bb.pb[PIECE_N] | bb.pb[PIECE_R]
    else:
        knights = bb.pw[PIECE_N] | bb.pw[PIECE_R]

    if knights and (attacks_from_knight(ksq) & knights):
        return True

    # King attacks (opposing king adjacency)
    if color == WHITE:
        kings = bb.pb[PIECE_K]
    else:
        kings = bb.pw[PIECE_K]

    if kings and (attacks_from_king(ksq) & kings):
        return True

    # Bishop attacks
    if color == WHITE:
        bishops = bb.pb[PIECE_B]
    else:
        bishops = bb.pw[PIECE_B]

    if bishops and (bishop_attacks_from(ksq, occ_all) & bishops):
        return True

    # Rook attacks
    if color == WHITE:
        rooks = bb.pb[PIECE_R]
    else:
        rooks = bb.pw[PIECE_R]

    if rooks and (rook_attacks_from(ksq, occ_all) & rooks):
        return True

    # Queen attacks (rook + bishop sliders)
    if color == WHITE:
        queens = bb.pb[PIECE_Q]
    else:
        queens = bb.pw[PIECE_Q]

    if queens:
        rook_att = rook_attacks_from(ksq, occ_all)
        bish_att = bishop_attacks_from(ksq, occ_all)
        if (rook_att | bish_att) & queens:
            return True
            
    # --- PARANOID VERIFICATION ---
    # User reported a bug where diagonal checks fail.
    # We run a manual raycast to be absolutely sure.
    if _in_check_slow(bb, color, ksq):
        return True

    return False

def _in_check_slow(bb: GameState, color: int, ksq: int) -> bool:
    # Manual raycast for sliding attacks to verify bitboards
    occ = bb.occ_all
    enemy = BLACK if color == WHITE else WHITE
    if enemy == WHITE:
        R = bb.pw[PIECE_R] | bb.pw[PIECE_Q]
        B = bb.pw[PIECE_B] | bb.pw[PIECE_Q]
    else:
        R = bb.pb[PIECE_R] | bb.pb[PIECE_Q]
        B = bb.pb[PIECE_B] | bb.pb[PIECE_Q]
        
    kx, ky = sq_to_xy(ksq)
    
    # Check Orthogonal (Rook/Queen)
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        cx, cy = kx+dx, ky+dy
        while in_board(cx, cy):
            sq = xy_to_sq(cx, cy)
            if occ & (1 << sq):
                if R & (1 << sq): return True
                break
            cx += dx
            cy += dy
            
    # Check Diagonal (Bishop/Queen)
    for dx, dy in ((1,1),(1,-1),(-1,1),(-1,-1)):
        cx, cy = kx+dx, ky+dy
        while in_board(cx, cy):
            sq = xy_to_sq(cx, cy)
            if occ & (1 << sq):
                if B & (1 << sq): return True
                break
            cx += dx
            cy += dy
            
    return False

def gives_check_fast(bb: GameState, m: Move) -> bool:
    """
    Determines if move 'm' gives check to the opponent.
    Avoiding step_forward().
    """
    color = (m >> 14) & 1 # MOVE_COLOR_SHIFT=14
    enemy_color = BLACK if color == WHITE else WHITE
    
    # Opponent King
    if enemy_color == WHITE:
        ksq = bb.wk_sq
    else:
        ksq = bb.bk_sq
        
    if ksq == -1: return False 
    
    piece = (m >> 11) & 7 # MOVE_PIECE_SHIFT=11
    to_sq = m & 0x1F      # MOVE_TO_MASK
    from_sq = (m >> 5) & 0x1F # MOVE_FROM_SHIFT
    
    pseudo_occ = (bb.occ_all & ~(1 << from_sq)) | (1 << to_sq)
    
    # 1. DIRECT CHECK
    if piece == PIECE_P:
        if attacks_from_pawn(color, to_sq) & (1 << ksq):
            return True
    elif piece == PIECE_N:
        if attacks_from_knight(to_sq) & (1 << ksq):
            return True
    elif piece == PIECE_K:
         pass 
    else:
        # Check promotion (acts as Queen)
        is_promo = (m >> 18) & 1
        eff_piece = PIECE_Q if is_promo else piece
            
        if eff_piece == PIECE_B:
            if bishop_attacks_from(to_sq, pseudo_occ) & (1 << ksq): return True
        elif eff_piece == PIECE_R:
            if (rook_attacks_from(to_sq, pseudo_occ) | attacks_from_knight(to_sq)) & (1 << ksq): return True
        elif eff_piece == PIECE_Q:
             if (bishop_attacks_from(to_sq, pseudo_occ) | rook_attacks_from(to_sq, pseudo_occ)) & (1 << ksq):
                 return True

    # 2. DISCOVERED CHECK
    exclude_mask = ~(1 << from_sq)
    if color == WHITE:
        my_B = (bb.pw[PIECE_B] | bb.pw[PIECE_Q]) & exclude_mask
        my_R = (bb.pw[PIECE_R] | bb.pw[PIECE_Q]) & exclude_mask
    else:
        my_B = (bb.pb[PIECE_B] | bb.pb[PIECE_Q]) & exclude_mask
        my_R = (bb.pb[PIECE_R] | bb.pb[PIECE_Q]) & exclude_mask
        
    if bishop_attacks_from(ksq, pseudo_occ) & my_B: return True
    if rook_attacks_from(ksq, pseudo_occ) & my_R: return True
        
    return False

def step_forward(state: GameState, move: Move, wtm: bool) -> Tuple[UndoInfo, bool]:
    """
    Fast make-move (no cloning).
    - Keeps scalar bitboards (WP..BK) and arrays (pw/pb) in sync
    - Recomputes occupancies from piece bitboards
    - Uses incremental eval (inc_eval_after_move)
    - Does NOT rely on zobrist; TT is effectively disabled by negamax/qsearch
    """
    color = WHITE if wtm else BLACK
    enemy = BLACK if color == WHITE else WHITE

    mover_piece = move_piece(move)
    from_sq     = move_from(move)
    to_sq       = move_to(move)
    is_cap      = move_is_capture(move)
    cap_type    = move_captured_piece(move)

    fr_bit = 1 << from_sq
    to_bit = 1 << to_sq

    # --- Save undo info (snapshots) ---
    undo = UndoInfo(
        piece=mover_piece,
        captured=cap_type,
        from_sq=from_sq,
        to_sq=to_sq,
        promoted=0,
        prev_zkey=state.zkey,      # we won't touch zkey incrementally, but keep it consistent
        prev_mg=state.mg,
        prev_eg=state.eg,
        prev_phase=state.phase,
        prev_occ_w=state.occ_w,
        prev_occ_b=state.occ_b,
        prev_occ=state.occ_all,
        prev_wtm=wtm,
        prev_mob_w=state.mob_w,
        prev_mob_b=state.mob_b,
        prev_wk_sq=state.wk_sq,
        prev_bk_sq=state.bk_sq,
    )

    # Append current key to history for repetition detection
    state.history.append(state.zkey)

    # Local refs to arrays
    pw = state.pw
    pb = state.pb

    # --- Helper: toggle piece bitboards (scalar + array) ---
    def _toggle_piece(color_, piece_, bit):
        if color_ == WHITE:
            pw[piece_] ^= bit
            if piece_ == PIECE_P: state.WP ^= bit
            elif piece_ == PIECE_N: state.WN ^= bit
            elif piece_ == PIECE_B: state.WB ^= bit
            elif piece_ == PIECE_R: state.WR ^= bit
            elif piece_ == PIECE_Q: state.WQ ^= bit
            elif piece_ == PIECE_K: 
                state.WK ^= bit
                # Update King Square Cache if moving (not just toggling for capture, etc... wait)
                # step_forward toggles 'from' then 'to'
                # But here we are inside _toggle_piece.
                # We should update caches explicitly outside, OR handle it here carefully.
                # Actually, easier to update explicitly in step_forward logic below.
        else:
            pb[piece_] ^= bit
            if piece_ == PIECE_P: state.BP ^= bit
            elif piece_ == PIECE_N: state.BN ^= bit
            elif piece_ == PIECE_B: state.BB ^= bit
            elif piece_ == PIECE_R: state.BR ^= bit
            elif piece_ == PIECE_Q: state.BQ ^= bit
            elif piece_ == PIECE_K: state.BK ^= bit

    # ----------------------------------------
    # 1) Remove mover from origin
    # ----------------------------------------
    _toggle_piece(color, mover_piece, fr_bit)
    state.zkey ^= Z_PIECE[color][mover_piece][from_sq]
    
    # Update King Cache if we moved the king
    if mover_piece == PIECE_K:
        if color == WHITE: state.wk_sq = -1 # Temporarily removed
        else: state.bk_sq = -1

    # ----------------------------------------
    # 2) Capture (remove enemy piece at 'to')
    # ----------------------------------------
    if is_cap and cap_type != CAPTURE_NONE:
        _toggle_piece(enemy, cap_type, to_bit)
        state.zkey ^= Z_PIECE[enemy][cap_type][to_sq]

    # ----------------------------------------
    # 3) Promotion check
    # ----------------------------------------
    promoted = 0
    if mover_piece == PIECE_P:
        # white promotes on rank 0 (sq < 5), black on rank 4 (sq >= 20)
        if (color == WHITE and to_sq < 5) or (color == BLACK and to_sq >= 20):
            promoted = 1
            undo.promoted = 1

    # ----------------------------------------
    # 4) Add new piece on destination
    # ----------------------------------------
    new_piece = PIECE_Q if promoted else mover_piece
    _toggle_piece(color, new_piece, to_bit)
    state.zkey ^= Z_PIECE[color][new_piece][to_sq]

    # Update King Cache if we placed the king
    if new_piece == PIECE_K:
        if color == WHITE: state.wk_sq = to_sq
        else: state.bk_sq = to_sq

    # ----------------------------------------
    # 5) Recompute occupancies from bitboards
    # ----------------------------------------
    state.occ_w = state.WP | state.WN | state.WB | state.WR | state.WQ | state.WK
    state.occ_b = state.BP | state.BN | state.BB | state.BR | state.BQ | state.BK
    state.occ_all = state.occ_w | state.occ_b

    # ----------------------------------------
    # 6) Incremental eval update
    # ----------------------------------------
    inc_eval_after_move(
        state,
        color,
        mover_piece,
        from_sq,
        to_sq,
        cap_type,
        promoted
    )

    # Update side to move in hash
    state.zkey ^= Z_SIDE

    next_wtm = not wtm
    state.white_to_move = next_wtm
    return undo, next_wtm

def step_back(state: GameState, undo: UndoInfo) -> bool:
    """
    Undo step_forward():
    - Restores piece bitboards by inverting the changes made in step_forward
    - Restores occupancies, eval, and zkey from UndoInfo
    - Returns the side-to-move BEFORE the undone move (undo.prev_wtm)
    """
    # Pop history
    state.history.pop()

    mover    = undo.piece
    cap      = undo.captured
    fr       = undo.from_sq
    to       = undo.to_sq
    promoted = undo.promoted
    prev_wtm = undo.prev_wtm

    color = WHITE if prev_wtm else BLACK
    enemy = BLACK if color == WHITE else WHITE

    fr_bit = 1 << fr
    to_bit = 1 << to

    pw = state.pw
    pb = state.pb

    def _toggle_piece(color_, piece_, bit):
        if color_ == WHITE:
            pw[piece_] ^= bit
            if piece_ == PIECE_P: state.WP ^= bit
            elif piece_ == PIECE_N: state.WN ^= bit
            elif piece_ == PIECE_B: state.WB ^= bit
            elif piece_ == PIECE_R: state.WR ^= bit
            elif piece_ == PIECE_Q: state.WQ ^= bit
            elif piece_ == PIECE_K: state.WK ^= bit
        else:
            pb[piece_] ^= bit
            if piece_ == PIECE_P: state.BP ^= bit
            elif piece_ == PIECE_N: state.BN ^= bit
            elif piece_ == PIECE_B: state.BB ^= bit
            elif piece_ == PIECE_R: state.BR ^= bit
            elif piece_ == PIECE_Q: state.BQ ^= bit
            elif piece_ == PIECE_K: state.BK ^= bit

    # ----------------------------------------
    # 1) Remove piece from destination
    #    (the piece currently there is Q if promoted else original mover)
    # ----------------------------------------
    piece_at_to = PIECE_Q if promoted else mover
    _toggle_piece(color, piece_at_to, to_bit)

    # ----------------------------------------
    # 2) Restore captured piece (if any)
    # ----------------------------------------
    if cap != CAPTURE_NONE:
        _toggle_piece(enemy, cap, to_bit)

    # ----------------------------------------
    # 3) Put original mover back on from_sq (always its original type)
    # ----------------------------------------
    _toggle_piece(color, mover, fr_bit)

    # ----------------------------------------
    # 4) Restore eval, phase, occupancies, and zobrist from snapshot
    # ----------------------------------------
    state.mg    = undo.prev_mg
    state.eg    = undo.prev_eg
    state.phase = undo.prev_phase

    state.occ_w   = undo.prev_occ_w
    state.occ_b   = undo.prev_occ_b
    state.occ_all = undo.prev_occ
    
    state.mob_w   = undo.prev_mob_w
    state.mob_b   = undo.prev_mob_b
    
    state.wk_sq   = undo.prev_wk_sq
    state.bk_sq   = undo.prev_bk_sq

    state.zkey = undo.prev_zkey

    # Side to move before the undone move
    # Side to move before the undone move
    state.white_to_move = prev_wtm
    return prev_wtm

def get_captured_type(bb: GameState, color: int, sq: int) -> int:
    """
    Return piece index (0..5) of the captured piece on `sq`, or CAPTURE_NONE.
    color = side to move (attacker), so we look at opponent's piece array.
    """
    mask = 1 << sq

    if color == WHITE:
        pieces = bb.pb
    else:
        pieces = bb.pw

    # 6-piece loop, Cython will unroll nicely
    for p in range(6):
        if pieces[p] & mask:
            return p

    return CAPTURE_NONE


def generate_moves(bb: GameState, color: int) -> List[Move]:
    """
    Generate pseudo-legal moves for side `color` in position `bb`.
    Returns a Python list of packed Move ints.
    Designed to be Cython-friendly:
      - local references
      - no nested function calls in the hot loops (inlined encode)
      - simple bitboard iteration
    """
    if color == WHITE:
        own = bb.occ_w
        opp = bb.occ_b
        P_arr = bb.pw
    else:
        own = bb.occ_b
        opp = bb.occ_w
        P_arr = bb.pb

    occ_all = bb.occ_all
    moves = []
    
    # Pre-shifted values for common fields
    col_shift = color << 14
    
    # ------------------
    # Pawns
    # ------------------
    P = P_arr[PIECE_P]
    if P:
        piece_shift = PIECE_P << 11
        # Common parts: piece | color | capture_none
        base_quiet = piece_shift | col_shift | (CAPTURE_NONE << 15)
        
        if color == WHITE:
            dy = -1
        else:
            dy = 1

        b = P
        while b:
            frb = b & -b
            fr = frb.bit_length() - 1
            b &= b - 1

            x = fr % W
            y = fr // W
            ny = y + dy

            # Forward Moves
            if 0 <= ny < H:
                to = ny * W + x
                to_mask = 1 << to
                
                # Check promotion
                promo_shift = 0
                if (color == WHITE and ny == 0) or (color == BLACK and ny == 4):
                    # Promotion: set promoted bit (value 1)
                    promo_shift = 1 << 18

                # Quiet forward move
                if not (occ_all & to_mask):
                    # to | (fr << 5) | (0 << 10) | base_quiet | promo_shift
                    moves.append(to | (fr << 5) | base_quiet | promo_shift)

                # Left capture
                if x > 0:
                    t = ny * W + (x - 1)
                    t_mask = 1 << t
                    if opp & t_mask:
                        cap_type = get_captured_type(bb, color, t)
                        # to | (fr << 5) | (1 << 10) | piece | color | (cap_type << 15) | promo
                        moves.append(t | (fr << 5) | (1 << 10) | piece_shift | col_shift | (cap_type << 15) | promo_shift)

                # Right capture
                if x < W - 1:
                    t = ny * W + (x + 1)
                    t_mask = 1 << t
                    if opp & t_mask:
                        cap_type = get_captured_type(bb, color, t)
                        moves.append(t | (fr << 5) | (1 << 10) | piece_shift | col_shift | (cap_type << 15) | promo_shift)

    # ------------------
    # Knights
    # ------------------
    N = P_arr[PIECE_N]
    if N:
        piece_shift = PIECE_N << 11
        b = N
        while b:
            frb = b & -b
            fr = frb.bit_length() - 1
            b &= b - 1

            for t in PRE_KNIGHT[fr]:
                t_mask = 1 << t
                if own & t_mask:
                    continue

                if opp & t_mask:
                    cap_type = get_captured_type(bb, color, t)
                    moves.append(t | (fr << 5) | (1 << 10) | piece_shift | col_shift | (cap_type << 15))
                else:
                    moves.append(t | (fr << 5) | piece_shift | col_shift | (CAPTURE_NONE << 15))

    # ------------------
    # Bishops
    # ------------------
    B = P_arr[PIECE_B]
    if B:
        piece_shift = PIECE_B << 11
        b = B
        while b:
            frb = b & -b
            fr = frb.bit_length() - 1
            b &= b - 1

            att = bishop_attacks_from(fr, occ_all) & ~own
            a = att
            while a:
                lsb = a & -a
                t = lsb.bit_length() - 1
                a &= a - 1

                t_mask = 1 << t
                if opp & t_mask:
                    cap_type = get_captured_type(bb, color, t)
                    moves.append(t | (fr << 5) | (1 << 10) | piece_shift | col_shift | (cap_type << 15))
                else:
                    moves.append(t | (fr << 5) | piece_shift | col_shift | (CAPTURE_NONE << 15))

    # ------------------
    # Rooks (Right piece)
    # ------------------
    R = P_arr[PIECE_R]
    if R:
        piece_shift = PIECE_R << 11
        b = R
        while b:
            frb = b & -b
            fr = frb.bit_length() - 1
            b &= b - 1

            # Slider Part
            att = rook_attacks_from(fr, occ_all) & ~own
            a = att
            while a:
                lsb = a & -a
                t = lsb.bit_length() - 1
                a &= a - 1

                t_mask = 1 << t
                if opp & t_mask:
                    cap_type = get_captured_type(bb, color, t)
                    moves.append(t | (fr << 5) | (1 << 10) | piece_shift | col_shift | (cap_type << 15))
                else:
                    moves.append(t | (fr << 5) | piece_shift | col_shift | (CAPTURE_NONE << 15))

            # Knight Part
            for t in PRE_KNIGHT[fr]:
                t_mask = 1 << t
                if own & t_mask:
                    continue

                if opp & t_mask:
                    cap_type = get_captured_type(bb, color, t)
                    moves.append(t | (fr << 5) | (1 << 10) | piece_shift | col_shift | (cap_type << 15))
                else:
                    moves.append(t | (fr << 5) | piece_shift | col_shift | (CAPTURE_NONE << 15))

    # ------------------
    # Queen
    # ------------------
    Q = P_arr[PIECE_Q]
    if Q:
        piece_shift = PIECE_Q << 11
        b = Q
        while b:
            frb = b & -b
            fr = frb.bit_length() - 1
            b &= b - 1

            att = (rook_attacks_from(fr, occ_all) | bishop_attacks_from(fr, occ_all)) & ~own
            a = att
            while a:
                lsb = a & -a
                t = lsb.bit_length() - 1
                a &= a - 1

                t_mask = 1 << t
                if opp & t_mask:
                    cap_type = get_captured_type(bb, color, t)
                    moves.append(t | (fr << 5) | (1 << 10) | piece_shift | col_shift | (cap_type << 15))
                else:
                    moves.append(t | (fr << 5) | piece_shift | col_shift | (CAPTURE_NONE << 15))

    # ------------------
    # King
    # ------------------
    K = P_arr[PIECE_K]
    if K:
        piece_shift = PIECE_K << 11
        fr = (K & -K).bit_length() - 1
        
        # Use PRE_KING
        for t in PRE_KING[fr]:
            t_mask = 1 << t
            if own & t_mask:
                continue

            if opp & t_mask:
                cap_type = get_captured_type(bb, color, t)
                moves.append(t | (fr << 5) | (1 << 10) | piece_shift | col_shift | (cap_type << 15))
            else:
                 moves.append(t | (fr << 5) | piece_shift | col_shift | (CAPTURE_NONE << 15))

    return moves

# =============================================================================
# EVALUATION
# =============================================================================

def pawn_key(state) -> int:
    """
    Return pawn-only 64-bit zobrist key.
    Uses two small static tables.
    """
    wp = state.pw[PIECE_P]
    bp = state.pb[PIECE_P]

    h = 0

    # white pawns
    x = wp
    while x:
        lsb = x & -x
        sq = lsb.bit_length() - 1
        h ^= PAWN_Z_W[sq]
        x ^= lsb

    # black pawns
    x = bp
    while x:
        lsb = x & -x
        sq = lsb.bit_length() - 1
        h ^= PAWN_Z_B[sq]
        x ^= lsb

    return h

@dataclass
class PawnEntry:
    key: int
    mg: int
    eg: int

class PawnCache:
    __slots__ = ("size", "mask", "table", "hits", "probes")

    def __init__(self, size=1 << 15) -> None:  # 32k → perfect for 5×5
        self.size = size
        self.mask = size - 1
        self.table = [None] * size
        self.hits = 0
        self.probes = 0

    def probe(self, key: int) -> Optional[PawnEntry]:
        """Probe the cache for a pawn entry."""
        self.probes += 1
        e = self.table[key & self.mask]
        if e and e.key == key:
            self.hits += 1
            return e
        return None

    def store(self, key: int, mg: int, eg: int) -> None:
        """Store a pawn entry in the cache."""
        self.table[key & self.mask] = PawnEntry(key, mg, eg)

PAWN_STRUCTURE_CACHE = PawnCache()

def eval_pawns(bb) -> Tuple[int, int]:
    """
    Returns (mg_score, eg_score) for pawn structure.
    Uses PawnCache to avoid recomputation.
    """
    key = pawn_key(bb)
    entry = PAWN_STRUCTURE_CACHE.probe(key)
    if entry:
        return entry.mg, entry.eg

    mg = 0
    eg = 0

    WP = bb.pw[PIECE_P]
    BP = bb.pb[PIECE_P]

    wp_file = [0,0,0,0,0]
    bp_file = [0,0,0,0,0]

    # --- file counts ---
    x = WP
    while x:
        lsb = x & -x
        sq = lsb.bit_length() - 1
        file = sq % 5
        wp_file[file] += 1
        x ^= lsb

    x = BP
    while x:
        lsb = x & -x
        sq = lsb.bit_length() - 1
        file = sq % 5
        bp_file[file] += 1
        x ^= lsb

    # --- doubled / isolated ---
    for f in range(5):
        # white doubled
        if wp_file[f] >= 2:
            mg += EVAL_DOUBLED_MG
            eg += EVAL_DOUBLED_EG
        # white isolated
        if wp_file[f] > 0:
            if (f == 0 or wp_file[f-1] == 0) and (f == 4 or wp_file[f+1] == 0):
                mg += EVAL_ISOLATED_MG
                eg += EVAL_ISOLATED_EG

        # black doubled
        if bp_file[f] >= 2:
            mg -= EVAL_DOUBLED_MG
            eg -= EVAL_DOUBLED_EG

        # black isolated
        if bp_file[f] > 0:
            if (f == 0 or bp_file[f-1] == 0) and (f == 4 or bp_file[f+1] == 0):
                mg -= EVAL_ISOLATED_MG
                eg -= EVAL_ISOLATED_EG

    # --- passed pawns helpers ---
    def passed_white(sq):
        x = sq % 5
        y = sq // 5
        for fx in (x-1, x, x+1):
            if 0 <= fx < 5:
                # White moves y+, check y+1 to 4
                for ry in range(y+1, 5):
                    if BP & (1 << (ry*5+fx)):
                        return False
        return True

    def passed_black(sq):
        x = sq % 5
        y = sq // 5
        for fx in (x-1, x, x+1):
            if 0 <= fx < 5:
                # Black moves y-, check y-1 to 0
                for ry in range(y-1, -1, -1):
                    if WP & (1 << (ry*5+fx)):
                        return False
        return True

    # --- White Pawns ---
    x = WP
    while x:
        lsb = x & -x
        sq = lsb.bit_length() - 1
        file = sq % 5
        y = sq // 5
        rank = y # 0=base, 4=promo
        
        is_passed = passed_white(sq)
        
        if is_passed:
            # Rank Scaling
            mg += PASS_MG[rank]
            eg += PASS_EG[rank]
            
            # Protected (by friendly pawn)
            # Check if any WP attacks this square
            # White pawns at (x-1, y-1) or (x+1, y-1) attack (x, y)
            is_protected = False
            if y > 0:
                if file > 0 and (WP & (1 << ((y-1)*5 + file-1))): is_protected = True
                elif file < 4 and (WP & (1 << ((y-1)*5 + file+1))): is_protected = True
            
            if is_protected:
                mg += PROTECTED_PASS_MG
                eg += PROTECTED_PASS_EG
                
            # Connected (Friendly pawn on adjacent file, same rank or +/- 1)
            # Simplified: adjacent file, rank >= y-1
            is_connected = False
            for f in (file-1, file+1):
                if 0 <= f < 5:
                    # Check ranks y-1, y, y+1
                    if y > 0 and (WP & (1 << ((y-1)*5+f))): is_connected = True
                    if (WP & (1 << (y*5+f))): is_connected = True
                    if y < 4 and (WP & (1 << ((y+1)*5+f))): is_connected = True
            
            if is_connected:
                mg += CONNECTED_PASS_MG
                eg += CONNECTED_PASS_EG
                
            # Outside (Distance to enemy King)
            if bb.pb[PIECE_K]:
                ksq = (bb.pb[PIECE_K] & -bb.pb[PIECE_K]).bit_length() - 1
                kx = ksq % 5
                dist = abs(file - kx)
                if dist > 2: # Far away
                    eg += 30 # Outside bonus
                
                # [ENDGAME ONLY] Advanced Heuristics
                if bb.phase < 12:
                    ky = ksq // 5
                    
                    # 1. King Distance to Pawn
                    if bb.pw[PIECE_K]:
                        wksq = (bb.pw[PIECE_K] & -bb.pw[PIECE_K]).bit_length() - 1
                        wkx, wky = wksq % 5, wksq // 5
                        dist_wk = max(abs(file - wkx), abs(y - wky))
                        dist_bk = max(abs(file - kx), abs(y - ky))
                        
                        # Closer friendly king is better
                        eg += (dist_bk - dist_wk) * 10
                        
                    # 2. Promotion Square Danger
                    # Check distance of Enemy King to Promotion Square (file, 4)
                    dist_bk_promo = max(abs(file - kx), abs(4 - ky))
                    eg += dist_bk_promo * 15
                    
                    # 3. Blocked Passed Pawn Penalty
                    # Check square directly in front (y+1)
                    if y < 4:
                        front_sq = (y + 1) * 5 + file
                        if bb.occ_all & (1 << front_sq):
                            mg -= 20
                            eg -= 40
                    
        else:
            # Candidate / Backward
            # Candidate: No enemy pawn on file in front
            is_candidate = True
            for r in range(y + 1, 5): # Moving UP
                if BP & (1 << (r * 5 + file)):
                    is_candidate = False
                    break
            
            if is_candidate:
                mg += 10
                eg += 20
                
            # Backward
            pass 
            
        x ^= lsb

    # --- Black Pawns ---
    x = BP
    while x:
        lsb = x & -x
        sq = lsb.bit_length() - 1
        file = sq % 5
        y = sq // 5
        rank = 4 - y # 0=base, 4=promo
        
        is_passed = passed_black(sq)
        
        if is_passed:
            # Rank Scaling
            mg -= PASS_MG[rank]
            eg -= PASS_EG[rank]
            
            # Protected
            is_protected = False
            if y < 4:
                if file > 0 and (BP & (1 << ((y+1)*5 + file-1))): is_protected = True
                elif file < 4 and (BP & (1 << ((y+1)*5 + file+1))): is_protected = True
                
            if is_protected:
                mg -= PROTECTED_PASS_MG
                eg -= PROTECTED_PASS_EG
                
            # Connected
            is_connected = False
            for f in (file-1, file+1):
                if 0 <= f < 5:
                    if y > 0 and (BP & (1 << ((y-1)*5+f))): is_connected = True
                    if (BP & (1 << (y*5+f))): is_connected = True
                    if y < 4 and (BP & (1 << ((y+1)*5+f))): is_connected = True
            
            if is_connected:
                mg -= CONNECTED_PASS_MG
                eg -= CONNECTED_PASS_EG
                
            # Outside
            if bb.pw[PIECE_K]:
                ksq = (bb.pw[PIECE_K] & -bb.pw[PIECE_K]).bit_length() - 1
                kx = ksq % 5
                dist = abs(file - kx)
                if dist > 2:
                    eg -= 30
                    
                # [ENDGAME ONLY] Advanced Heuristics
                if bb.phase < 12:
                    ky = ksq // 5
                    
                    # 1. King Distance to Pawn
                    if bb.pb[PIECE_K]:
                        bksq = (bb.pb[PIECE_K] & -bb.pb[PIECE_K]).bit_length() - 1
                        bkx, bky = bksq % 5, bksq // 5
                        dist_bk = max(abs(file - bkx), abs(y - bky))
                        dist_wk = max(abs(file - kx), abs(y - ky))
                        
                        # Closer friendly king is better (negative score for black)
                        eg -= (dist_wk - dist_bk) * 10

                    # 2. Promotion Square Danger
                    # Check distance of Enemy King to Promotion Square (file, 0)
                    dist_wk_promo = max(abs(file - kx), abs(0 - ky))
                    eg -= dist_wk_promo * 15

                    # 3. Blocked Passed Pawn Penalty
                    if y > 0:
                        front_sq = (y - 1) * 5 + file
                        if bb.occ_all & (1 << front_sq):
                            mg += 20 
                            eg += 40
                    
        else:
            # Candidate
            is_candidate = True
            for r in range(y - 1, -1, -1): # Moving DOWN
                if WP & (1 << (r * 5 + file)):
                    is_candidate = False
                    break
            
            if is_candidate:
                mg -= 10
                eg -= 20
                
        x ^= lsb

    PAWN_STRUCTURE_CACHE.store(key, mg, eg)
    return mg, eg

# =============================================================================
# SIMPLE KING SAFETY / MOBILITY (CHEAP)
# =============================================================================







def exposed_file_risk_OLD(bb: GameState, kx: int, white: bool) -> int:
    """
    Penalize king on open/semi-open file with enemy heavies.
    """
    mask = FILE_MASKS[kx]
    if white:
        my_p = bb.pw[PIECE_P] & mask
        enemy_p = bb.pb[PIECE_P] & mask
        heavies = (bb.pb[PIECE_R] | bb.pb[PIECE_Q]) & mask
    else:
        my_p = bb.pb[PIECE_P] & mask
        enemy_p = bb.pw[PIECE_P] & mask
        heavies = (bb.pw[PIECE_R] | bb.pw[PIECE_Q]) & mask

    if not heavies:
        return 0

    # Open file = no pawns at all
    if not my_p and not enemy_p:
        return -10
    # Semi-open = no friendly pawn
    if not my_p:
        return -6

    return 0

def adjacent_ring_mask(x: int, y: int) -> int:
    """Return bitmask of squares surrounding (x, y)."""
    mask = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < 5 and 0 <= ny < 5:
                mask |= (1 << (ny * 5 + nx))
    return mask

def pawn_screen_value(bb: GameState, kx: int, ky: int, white: bool) -> int:
    """Calculate bonus for pawns shielding the king."""
    # Bonus for pawns in front of king
    score = 0
    pawns = bb.pw[PIECE_P] if white else bb.pb[PIECE_P]
    
    # Direction "forward"
    dy = -1 if white else 1
    
    # Check 3 files around king
    for dx in (-1, 0, 1):
        nx = kx + dx
        if 0 <= nx < 5:
            # Check immediate square in front
            ny = ky + dy
            if 0 <= ny < 5:
                sq = ny * 5 + nx
                if pawns & (1 << sq):
                    score += 10
                else:
                    # Check one more step (if 5x5 allows)
                    ny2 = ny + dy
                    if 0 <= ny2 < 5:
                        sq2 = ny2 * 5 + nx
                        if pawns & (1 << sq2):
                            score += 5
    return score



def exposed_file_risk(bb: GameState, kx: int, white: bool) -> int:
    """Calculate penalty for open/semi-open files near king."""
    # Penalty for open/semi-open files around king
    score = 0
    # Check files kx-1, kx, kx+1
    for x in range(max(0, kx-1), min(5, kx+2)):
        f_mask = 0
        for r in range(5):
            f_mask |= (1 << (r*5 + x))
            
        my_pawns = bb.pw[PIECE_P] if white else bb.pb[PIECE_P]
        enemy_pawns = bb.pb[PIECE_P] if white else bb.pw[PIECE_P]
        
        has_my = (my_pawns & f_mask)
        has_enemy = (enemy_pawns & f_mask)
        
        if not has_my:
            score -= 25 # Semi-open (Increased from 15)
            if not has_enemy:
                score -= 25 # Fully open (total -50) (Increased from 15)
                
    return score

def threats_into_region(bb: GameState, mask: int, enemy_color: int) -> dict:
    """Count enemy attackers targeting a specific region mask."""
    # Count attackers of each type hitting the mask
    counts = {"N": 0, "B": 0, "R": 0, "Q": 0}
    
    # Enemy pieces
    if enemy_color == WHITE:
        N = bb.pw[PIECE_N]
        B = bb.pw[PIECE_B]
        R = bb.pw[PIECE_R]
        Q = bb.pw[PIECE_Q]
        occ = bb.occ_all
    else:
        N = bb.pb[PIECE_N]
        B = bb.pb[PIECE_B]
        R = bb.pb[PIECE_R]
        Q = bb.pb[PIECE_Q]
        occ = bb.occ_all
        
    # We need to check if any enemy piece attacks any square in 'mask'
    # This is expensive to loop all squares in mask.
    # Better: Loop all enemy pieces and check if their attack set intersects mask.
    
    # Knights
    x = N
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        att = PRE_KNIGHT[sq]
        if att_mask_from_list(att) & mask:
            counts["N"] += 1
            
    # bishops
    x = B
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        att = bishop_attacks_from(sq, occ)
        if att & mask:
            counts["B"] += 1

    # rooks (right piece rook part)
    x = R
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        att = rook_attacks_from(sq, occ)
        if att & mask:
            counts["R"] += 1

    # queens
    x = Q
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        att = (bishop_attacks_from(sq, occ) | rook_attacks_from(sq, occ))
        if att & mask:
            counts["Q"] += 1
            
    return counts

def att_mask_from_list(sq_list: List[int]) -> int:
    """Convert list of squares to bitmask."""
    m = 0
    for s in sq_list:
        m |= (1 << s)
    return m

def eval_threats(bb: GameState, color: int) -> int:
    """Calculate penalties for pieces under attack."""
    # Penalize hanging pieces and pieces attacked by lower value
    score = 0
    
    if color == WHITE:
        my_pieces = bb.occ_w
        enemy_color = BLACK
        my_p = bb.pw[PIECE_P]
        my_n = bb.pw[PIECE_N]
        my_b = bb.pw[PIECE_B]
        my_r = bb.pw[PIECE_R]
        my_q = bb.pw[PIECE_Q]
    else:
        my_pieces = bb.occ_b
        enemy_color = WHITE
        my_p = bb.pb[PIECE_P]
        my_n = bb.pb[PIECE_N]
        my_b = bb.pb[PIECE_B]
        my_r = bb.pb[PIECE_R]
        my_q = bb.pb[PIECE_Q]
        
    # For each of my pieces, check if attacked
    # Check all pieces for threats
    targets = my_pieces
    
    while targets:
        sq = (targets & -targets).bit_length() - 1
        targets &= targets - 1
        
        # Get lowest value attacker
        if color == WHITE:
            op, on, ob, or_, oq, ok = bb.pb[PIECE_P], bb.pb[PIECE_N], bb.pb[PIECE_B], bb.pb[PIECE_R], bb.pb[PIECE_Q], bb.pb[PIECE_K]
        else:
            op, on, ob, or_, oq, ok = bb.pw[PIECE_P], bb.pw[PIECE_N], bb.pw[PIECE_B], bb.pw[PIECE_R], bb.pw[PIECE_Q], bb.pw[PIECE_K]
            
        att_sq, att_val = get_lowest_attacker(bb, sq, enemy_color, bb.occ_all, op, on, ob, or_, oq, ok)
        
        if att_sq != -1:
            # We are attacked!
            # Value of our piece
            my_val = 0
            if (1 << sq) & my_q: my_val = 900
            elif (1 << sq) & my_r: my_val = 500 # Rook or Right
            elif (1 << sq) & my_b: my_val = 300
            elif (1 << sq) & my_n: my_val = 300
            elif (1 << sq) & my_p: my_val = 100
            
            # If attacker is lower value, big penalty (Hanging to lower)
            if att_val < my_val:
                score -= (my_val - att_val) * 2
            else:
                # Attacked by equal/higher.
                # If not defended -> Hanging!
                # Simplified: penalize being attacked
                score -= 50 # Increased from 30
                
    return score

def file_tension_eval(bb: GameState, kx: int, white: bool) -> int:
    """Calculate penalty for heavies on the same file as the king."""
    pen = 0
    if white:
        R = bb.pb[PIECE_R]
        Q = bb.pb[PIECE_Q]
    else:
        R = bb.pw[PIECE_R]
        Q = bb.pw[PIECE_Q]

    x = R
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        if (sq % 5) == kx:
            pen -= 2

    x = Q
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        if (sq % 5) == kx:
            pen -= 4

    return pen

def king_safety(bb: GameState, color: int) -> int:
    """
    Aggregates king safety heuristics.
    Returns a score (usually negative penalty or small positive bonus).
    """
    if color == WHITE:
        K = bb.pw[PIECE_K]
        is_white = True
    else:
        K = bb.pb[PIECE_K]
        is_white = False

    if not K:
        return 0

    ksq = (K & -K).bit_length() - 1
    kx = ksq % 5
    ky = ksq // 5

    score = 0
    
    # 1. Pawn Shield
    score += pawn_screen_value(bb, kx, ky, is_white)

    # 2. File Exposure
    score += exposed_file_risk(bb, kx, is_white)

    # 3. File Tension (Heavies on file)
    score += file_tension_eval(bb, kx, is_white)

    # 4. Attackers around king
    ring = adjacent_ring_mask(kx, ky)
    # Fix: pass correct enemy color (BLACK if is_white else WHITE)
    # previously passed 'not is_white' which evaluated to False (0=WHITE) when is_white=True
    threats = threats_into_region(bb, ring, BLACK if is_white else WHITE)
    
    # 5. King Activity (Midgame)
    # Penalize King in corners or back rank if phase is Midgame (implied by high piece count)
    # We don't have phase passed in here, but we can check piece count or just apply generally.
    # In 5x5, King should be centralizing earlier.
    # Corners: 0, 4, 20, 24.
    if ksq in [0, 4, 20, 24]:
        score -= 5
        
    # Back rank penalty (if not castled/safe)
    # White back rank y=4. Black y=0.
    # Back rank penalty (if not castled/safe)
    # White back rank y=4. Black y=0.
    if is_white:
        if ky == 4: score -= 2
    else:
        if ky == 0: score -= 2

    # Weighted penalty for attackers
    # N=20, B=20, R=40, Q=80
    attack_weight = (threats["N"] * 20 + 
                     threats["B"] * 20 + 
                     threats["R"] * 40 + 
                     threats["Q"] * 80)
    
    # Non-linear scaling: more attackers = much worse
    # Cap the penalty to avoid evaluation explosion (e.g. -2000)
    # Max penalty should be around -300 (equivalent to a minor piece)
    total_penalty = min(attack_weight, 300)
    score -= total_penalty
    return score

# =============================================================================
# STRATEGIC EVALUATION HELPERS
# =============================================================================

def mobility(bb: GameState, color: int) -> int:
    """
    Calculate mobility score with granular weights per piece type.
    """
    score = 0
    occ = bb.occ_all
    
    if color == WHITE:
        my_n = bb.pw[PIECE_N]
        my_b = bb.pw[PIECE_B]
        my_r = bb.pw[PIECE_R]
        my_q = bb.pw[PIECE_Q]
        my_ri = bb.pw[PIECE_R] # Right piece is stored in R for now? No, it has its own.
        # Wait, PIECE_R is index 3. PIECE_Q is 4. PIECE_K is 5.
        # Where is Right piece?
        # In agent.py, PIECE_R is "Right" (Rook+Knight).
        # Wait, let's check constants.
        # PIECE_R = 3 (Right/Rook+Knight).
        # There is no separate Rook.
        # The game variant is 5x5.
        # Pieces: P, N, B, R (Right), Q, K.
        # So "R" IS the "Right" piece.
        # It moves as Rook + Knight.
        pass
    else:
        my_n = bb.pb[PIECE_N]
        my_b = bb.pb[PIECE_B]
        my_r = bb.pb[PIECE_R]
        my_q = bb.pb[PIECE_Q]

    # Weights
    # Weights (Reduced for 5x5)
    W_N = 1
    W_B = 1
    W_R = 2 # Right piece is powerful but board is small
    W_Q = 1

    # Knights
    x = my_n
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        moves = attacks_from_knight(sq) & ~occ # Pseudo-legal moves (captures included in ~occ? No, ~own)
        # Mobility usually counts squares you can move to.
        # ~occ excludes ALL pieces. Captures are moves.
        # So we should use ~own_pieces.
        # But for speed, ~occ is often used for "safe" mobility.
        # Let's use ~own.
        if color == WHITE:
            moves = attacks_from_knight(sq) & ~bb.occ_w
        else:
            moves = attacks_from_knight(sq) & ~bb.occ_b
        score += moves.bit_count() * W_N

    # Bishops
    x = my_b
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        if color == WHITE:
            moves = bishop_attacks_from(sq, occ) & ~bb.occ_w
        else:
            moves = bishop_attacks_from(sq, occ) & ~bb.occ_b
        score += moves.bit_count() * W_B

    # Right Pieces (Rook + Knight)
    x = my_r
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        if color == WHITE:
            # Rook moves
            r_moves = rook_attacks_from(sq, occ)
            # Knight moves
            n_moves = attacks_from_knight(sq)
            moves = (r_moves | n_moves) & ~bb.occ_w
        else:
            r_moves = rook_attacks_from(sq, occ)
            n_moves = attacks_from_knight(sq)
            moves = (r_moves | n_moves) & ~bb.occ_b
        score += moves.bit_count() * W_R

    # Queens
    x = my_q
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        if color == WHITE:
            moves = (bishop_attacks_from(sq, occ) | rook_attacks_from(sq, occ)) & ~bb.occ_w
        else:
            moves = (bishop_attacks_from(sq, occ) | rook_attacks_from(sq, occ)) & ~bb.occ_b
        score += moves.bit_count() * W_Q

    return score



def eval_queen_danger(bb: GameState, color: int) -> int:
    """
    Early game: Penalty if Queen is out and exposed.
    """
    score = 0
    # If phase is high (early game)
    if bb.phase > 16:
        if color == WHITE:
            Q = bb.pw[PIECE_Q]
            # If Queen is not on back rank (y=4)
            if Q and not (Q & (0b11111 << 20)):
                # Queen is out.
                # Check if attacked by minor pieces?
                # That's in eval_threats.
                # Just penalize early development slightly?
                score -= 10
        else:
            Q = bb.pb[PIECE_Q]
            if Q and not (Q & 0b11111):
                score -= 10
                
    return score

def eval_fork_potential(bb: GameState, color: int) -> int:
    """
    Detect if Knight/Right piece is attacking 2+ valuable pieces.
    """
    score = 0
    if color == WHITE:
        knights = bb.pw[PIECE_N] | bb.pw[PIECE_R] # Right acts as Knight
        targets = bb.pb[PIECE_Q] | bb.pb[PIECE_R] | bb.pb[PIECE_K]
    else:
        knights = bb.pb[PIECE_N] | bb.pb[PIECE_R]
        targets = bb.pw[PIECE_Q] | bb.pw[PIECE_R] | bb.pw[PIECE_K]
        
    x = knights
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        
        # Attacks
        att = attacks_from_knight(sq)
        
        # Count valuable targets hit
        hit = att & targets
        count = hit.bit_count()
        
        if count >= 2:
            score += 40 # Fork!
            
    return score


# =============================================================================
# EVALUATION (TAPERED)
# =============================================================================


# ===========================================================
# Pawn structure evaluation constants (5×5 tuned values)
# ===========================================================

# Doubled pawns penalty (MG & EG)
# Doubled pawns penalty (MG & EG)
# Defined at top

# Isolated pawn penalty
# Defined at top

# Backward pawn penalty
# Defined at top

# Passed pawn bonus by rank (0..4)
# rank 0 = promotion rank (should not happen for pawn on board?)
# rank 1 = 7th rank
# rank 2 = 6th rank
# rank 3 = 5th rank
# rank 4 = 4th rank (start)
# We index by "rank from promotion" (0 is promo, 4 is start)
# But in pawn_eval we used r = 4-y (white) or y (black).
# So r=0 is start (y=4 for white), r=4 is promo (y=0 for white).
# Wait, in pawn_eval I used:
# r = 4 - y (White). If y=4 (start), r=0. If y=0 (promo), r=4.
# So index 0 is start, index 4 is promo.
# Values should increase with index.
PASS_MG = [0, 10, 30, 60, 100]
PASS_EG = [0, 20, 50, 90, 140]

# Connected passed pawns bonus
CONNECTED_PASS = 20


# Advanced Evaluation Constants
KNIGHT_OUTPOST_BONUS = 15  # Was 25
ROOK_ON_7TH_BONUS = 20     # Was 30
BAD_BISHOP_PENALTY = -10   # Was -15

def eval_center_control(bb: GameState, color: int) -> int:
    """Reward control of center squares and inner ring."""
    # Center: (2,2)
    # Inner Ring: (1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)
    
    score = 0
    
    # Define center masks
    CENTER_SQ = 1 << 12 # (2,2) -> 2 + 2*5 = 12
    INNER_RING = 0x1CE738 # (1,1)..(3,3) excluding (2,2)
    
    center_mask = (1 << 12)
    inner_ring_mask = 0
    for x in range(1, 4):
        for y in range(1, 4):
            if x == 2 and y == 2: continue
            inner_ring_mask |= (1 << (y * 5 + x))
            
    if color == WHITE:
        my_occ = bb.occ_w
        pawns = bb.pw[PIECE_P]
        # Pawn attacks
        p_att = 0
        x = pawns
        while x:
            sq = (x & -x).bit_length() - 1
            x &= x - 1
            px, py = sq % 5, sq // 5
            if px > 0 and py > 0: p_att |= (1 << ((py - 1) * 5 + px - 1))
            if px < 4 and py > 0: p_att |= (1 << ((py - 1) * 5 + px + 1))
    else:
        my_occ = bb.occ_b
        pawns = bb.pb[PIECE_P]
        p_att = 0
        x = pawns
        while x:
            sq = (x & -x).bit_length() - 1
            x &= x - 1
            px, py = sq % 5, sq // 5
            if px > 0 and py < 4: p_att |= (1 << ((py + 1) * 5 + px - 1))
            if px < 4 and py < 4: p_att |= (1 << ((py + 1) * 5 + px + 1))

    # Occupation bonuses
    occ_center = (my_occ & center_mask)
    if occ_center: score += 25 # Increased from 15
    
    occ_ring = (my_occ & inner_ring_mask)
    score += bin(occ_ring).count('1') * 10 # Increased from 5
    
    # Pawn control bonuses (Reduced slightly to favor pieces)
    p_center = (p_att & center_mask)
    if p_center: score += 15 # Reduced from 20
    
    p_ring = (p_att & inner_ring_mask)
    score += bin(p_ring).count('1') * 5 # Reduced from 10
    
    return score

def eval_development(bb: GameState, color: int) -> int:
    """Reward minor pieces moving off the back rank."""
    score = 0
    if color == WHITE:
        back_rank = 4
        pieces = bb.pw[PIECE_N] | bb.pw[PIECE_B] | bb.pw[PIECE_R]
    else:
        back_rank = 0
        pieces = bb.pb[PIECE_N] | bb.pb[PIECE_B] | bb.pb[PIECE_R]
        
    while pieces:
        sq = (pieces & -pieces).bit_length() - 1
        pieces &= pieces - 1
        y = sq // 5
        if y != back_rank:
            score += 40 # Increased from 25
            
    return score

def eval_trapped_right(bb: GameState, color: int) -> int:
    """
    Penalty if the Right piece (Rook+Knight) is in a corner and blocked.
    """
    score = 0
    if color == WHITE:
        R = bb.pw[PIECE_R]
        occ = bb.occ_all # All pieces block
    else:
        R = bb.pb[PIECE_R]
        occ = bb.occ_all
        
    if not R: return 0
    
    # Check corners: 0, 4, 20, 24
    # If Right is in corner
    if R & 0x1100011: # Mask for corners
        sq = (R & -R).bit_length() - 1
        # Check mobility briefly
        # Knight moves
        n_moves = attacks_from_knight(sq)
        # Rook moves
        r_moves = rook_attacks_from(sq, occ)
        
        moves = (n_moves | r_moves) & ~occ # Capture or move
        # Wait, ~occ means empty squares. Captures are valid moves.
        # So & ~own_pieces.
        if color == WHITE: moves &= ~bb.occ_w
        else: moves &= ~bb.occ_b
        
        count = bin(moves).count('1')
        if count <= 2:
            score -= 20 # Trapped!
            
    return score

def eval_coordination(bb: GameState, color: int) -> int:
    """
    Bonus if Queen or Right piece is protected by another friendly piece.
    """
    score = 0
    if color == WHITE:
        targets = bb.pw[PIECE_Q] | bb.pw[PIECE_R]
        my_occ = bb.occ_w
        enemy_color = BLACK # We want to check if *White* attacks targets.
        # So we use get_lowest_attacker with attacker_color=WHITE
        # But get_lowest_attacker expects enemy_color to be the attacker?
        # Let's check get_lowest_attacker signature.
        # get_lowest_attacker(bb, sq, attacker_color, occ, ...)
        # We want to know if WHITE attacks sq.
        attacker_color = WHITE
        op, on, ob, or_, oq, ok = bb.pw[PIECE_P], bb.pw[PIECE_N], bb.pw[PIECE_B], bb.pw[PIECE_R], bb.pw[PIECE_Q], bb.pw[PIECE_K]
    else:
        targets = bb.pb[PIECE_Q] | bb.pb[PIECE_R]
        my_occ = bb.occ_b
        attacker_color = BLACK
        op, on, ob, or_, oq, ok = bb.pb[PIECE_P], bb.pb[PIECE_N], bb.pb[PIECE_B], bb.pb[PIECE_R], bb.pb[PIECE_Q], bb.pb[PIECE_K]
        
    x = targets
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        
        # Check if protected by friendly
        # We need to exclude the piece itself from attackers?
        # get_lowest_attacker checks if *any* piece of attacker_color attacks sq.
        # If sq contains a piece of attacker_color, does it attack itself?
        # No, ray casts usually skip self or start from self.
        # But Knight attacks don't include self.
        # So it should be fine.
        
        att_sq, _ = get_lowest_attacker(bb, sq, attacker_color, bb.occ_all, op, on, ob, or_, oq, ok)
        if att_sq != -1:
            score += 10 # Protected!
            
    return score



def eval_mopup(bb: GameState, color: int, my_material: int, opp_material: int) -> int:
    """
    Encourage driving the enemy king to the edge and bringing our king closer
    when we have a significant material advantage.
    """
    if my_material <= opp_material + 200:
        return 0
        
    score = 0
    
    # Locate Kings
    if color == WHITE:
        my_k = bb.pw[PIECE_K]
        opp_k = bb.pb[PIECE_K]
    else:
        my_k = bb.pb[PIECE_K]
        opp_k = bb.pw[PIECE_K]
        
    if not my_k or not opp_k:
        return 0
        
    my_ksq = (my_k & -my_k).bit_length() - 1
    opp_ksq = (opp_k & -opp_k).bit_length() - 1
    
    my_x, my_y = sq_to_xy(my_ksq)
    opp_x, opp_y = sq_to_xy(opp_ksq)
    
    # 1. Drive enemy king to edge (Center Manhattan Distance)
    # Center is (2,2). Distance from center = abs(x-2) + abs(y-2).
    # Max distance is 2+2=4 (corners). Min is 0.
    # We want to MAXIMIZE this distance.
    dist_from_center = abs(opp_x - 2) + abs(opp_y - 2)
    score += dist_from_center * 10
    
    # 2. Bring our king closer (Chebyshev Distance)
    # dist = max(abs(dx), abs(dy))
    # We want to MINIMIZE this distance.
    dist_between_kings = max(abs(my_x - opp_x), abs(my_y - opp_y))
    score += (4 - dist_between_kings) * 10
    
    return score

def eval_threats(bb: GameState, color: int) -> int:
    """
    Returns negative score if pieces are under attack (hanging).
    """
    score = 0
    if color == WHITE:
        my_pieces = bb.pw[PIECE_P] | bb.pw[PIECE_N] | bb.pw[PIECE_B] | bb.pw[PIECE_R] | bb.pw[PIECE_Q]
        enemy_color = BLACK
        op, on, ob, or_, oq, ok = bb.pb[PIECE_P], bb.pb[PIECE_N], bb.pb[PIECE_B], bb.pb[PIECE_R], bb.pb[PIECE_Q], bb.pb[PIECE_K]
    else:
        my_pieces = bb.pb[PIECE_P] | bb.pb[PIECE_N] | bb.pb[PIECE_B] | bb.pb[PIECE_R] | bb.pb[PIECE_Q]
        enemy_color = WHITE
        op, on, ob, or_, oq, ok = bb.pw[PIECE_P], bb.pw[PIECE_N], bb.pw[PIECE_B], bb.pw[PIECE_R], bb.pw[PIECE_Q], bb.pw[PIECE_K]
        
    occ = bb.occ_all
    
    x = my_pieces
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        
        # Check if attacked
        att, val = get_lowest_attacker(bb, sq, enemy_color, occ, op, on, ob, or_, oq, ok)
        if att != -1:
            score -= 50
            
    return score

def eval_queen_check_danger(bb: GameState, color: int) -> int:
    """
    Penalty if enemy Queen can check the King.
    """
    score = 0
    if color == WHITE:
        K = bb.pw[PIECE_K]
        Q = bb.pb[PIECE_Q]
        occ = bb.occ_all
    else:
        K = bb.pb[PIECE_K]
        Q = bb.pw[PIECE_Q]
        occ = bb.occ_all
        
    if not K or not Q: return 0
    
    ksq = (K & -K).bit_length() - 1
    qsq = (Q & -Q).bit_length() - 1
    
    k_ring = attacks_from_king(ksq)
    q_attacks = rook_attacks_from(qsq, occ) | bishop_attacks_from(qsq, occ)
    
    if k_ring & q_attacks:
        score -= 30
        
    return score

def eval_king_mobility(bb: GameState, color: int) -> int:
    """
    Calculate King's safe squares. Huge penalty if 0-1 safe squares.
    """
    score = 0
    if color == WHITE:
        K = bb.pw[PIECE_K]
        occ = bb.occ_all
        # Enemy attacks
        # We need to compute all squares attacked by Black.
        # This is expensive. We usually don't compute full attack map.
        # But for King Safety it's critical.
        # Let's approximate or compute just around the King.
        enemy_color = BLACK
        op, on, ob, or_, oq, ok = bb.pb[PIECE_P], bb.pb[PIECE_N], bb.pb[PIECE_B], bb.pb[PIECE_R], bb.pb[PIECE_Q], bb.pb[PIECE_K]
    else:
        K = bb.pb[PIECE_K]
        occ = bb.occ_all
        enemy_color = WHITE
        op, on, ob, or_, oq, ok = bb.pw[PIECE_P], bb.pw[PIECE_N], bb.pw[PIECE_B], bb.pw[PIECE_R], bb.pw[PIECE_Q], bb.pw[PIECE_K]
        
    if not K: return 0
    ksq = (K & -K).bit_length() - 1
    
    # King moves
    moves = attacks_from_king(ksq) & ~occ # Pseudo-legal moves (empty or capture)
    # Actually, we care about "safe" squares.
    # Even if occupied by friend, is it safe? (Flight square if friend moves?)
    # Usually we count pseudo-legal moves that are NOT attacked.
    
    # Filter moves that are attacked
    safe_count = 0
    x = moves
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        
        # Is 'sq' attacked by enemy?
        # We use get_lowest_attacker logic but just boolean.
        # is_square_attacked(bb, sq, attacker_color)
        # We don't have is_square_attacked exposed efficiently?
        # We can use get_lowest_attacker.
        att, _ = get_lowest_attacker(bb, sq, enemy_color, occ, op, on, ob, or_, oq, ok)
        if att == -1:
            safe_count += 1
            
    # Penalty
    # Penalty (Reduced for 5x5 opening)
    if safe_count == 0:
        score -= 100 # Danger! Mated or stuck.
    elif safe_count == 1:
        score -= 50 # Very restricted.
    elif safe_count == 2:
        score -= 10
        
    return score

def eval_queen_check_danger(bb: GameState, color: int) -> int:
    """
    Penalty if enemy Queen can give check on the next move.
    """
    score = 0
    if color == WHITE:
        # Check if BLACK Queen can check WHITE King
        Q = bb.pb[PIECE_Q]
        K = bb.pw[PIECE_K]
        occ = bb.occ_all
        my_occ = bb.occ_b # Black's occupancy (Queen's friends)
    else:
        Q = bb.pw[PIECE_Q]
        K = bb.pb[PIECE_K]
        occ = bb.occ_all
        my_occ = bb.occ_w
        
    if not Q or not K: return 0
    
    qsq = (Q & -Q).bit_length() - 1
    ksq = (K & -K).bit_length() - 1
    
    # 1. Generate squares from which Queen would check King
    # This is effectively "Queen attacks from King's square"
    # (Reverse: if Queen was at X, would it attack King?)
    # Since Queen attacks are symmetric, it's attacks_from_queen(ksq).
    check_squares = (rook_attacks_from(ksq, occ) | bishop_attacks_from(ksq, occ))
    
    # 2. Generate actual moves for the Queen
    # attacks_from_queen(qsq) & ~my_occ
    queen_moves = (rook_attacks_from(qsq, occ) | bishop_attacks_from(qsq, occ)) & ~my_occ
    
    # 3. Intersection
    can_check = check_squares & queen_moves
    
    if can_check:
        score -= 100 # High danger of check
        
    return score

def eval_pawn_mate_support(bb: GameState, color: int) -> int:
    """
    Penalty if enemy pawns control squares adjacent to the King (supporting mate).
    """
    score = 0
    if color == WHITE:
        K = bb.pw[PIECE_K]
        enemy_pawns = bb.pb[PIECE_P]
        # Black pawns attack "down" (y+1).
        # If Black pawn is at (x, y), it attacks (x-1, y+1) and (x+1, y+1).
        # We want to know if King's adjacent squares are attacked.
    else:
        K = bb.pb[PIECE_K]
        enemy_pawns = bb.pw[PIECE_P]
        # White pawns attack "up" (y-1).
        
    if not K: return 0
    ksq = (K & -K).bit_length() - 1
    
    # King's adjacent squares (mask)
    k_ring = attacks_from_king(ksq)
    
    # Squares attacked by enemy pawns
    p_att = 0
    x = enemy_pawns
    while x:
        sq = (x & -x).bit_length() - 1
        x &= x - 1
        px, py = sq % 5, sq // 5
        
        if color == WHITE: # Enemy is Black (attacks y-1)
            # Black pawns move down (y decreases), so they attack y-1
            if px > 0 and py > 0: p_att |= (1 << ((py - 1) * 5 + px - 1))
            if px < 4 and py > 0: p_att |= (1 << ((py - 1) * 5 + px + 1))
        else: # Enemy is White (attacks y+1)
            # White pawns move up (y increases), so they attack y+1
            if px > 0 and py < 4: p_att |= (1 << ((py + 1) * 5 + px - 1))
            if px < 4 and py < 4: p_att |= (1 << ((py + 1) * 5 + px + 1))
            
    # Intersection
    danger_squares = k_ring & p_att
    count = bin(danger_squares).count('1')
    
    if count > 0:
        score -= 50 * count # Dangerous pawn support near King
        
    return score

def detect_trivial_endgames(bb: GameState) -> Optional[int]:
    """
    Detects trivial wins that evaluation might miss.
    Returns Score (White perspective) or None.
    """
    # Check if opponent has only King
    w_pieces = bb.pw[PIECE_P] | bb.pw[PIECE_N] | bb.pw[PIECE_B] | bb.pw[PIECE_R] | bb.pw[PIECE_Q]
    b_pieces = bb.pb[PIECE_P] | bb.pb[PIECE_N] | bb.pb[PIECE_B] | bb.pb[PIECE_R] | bb.pb[PIECE_Q]
    
    if not b_pieces: # Black has only King
        # White has Major? (Q or R)
        if (bb.pw[PIECE_Q] or bb.pw[PIECE_R]):
            return 20000 # Winning score
        
        # [ENDGAME ONLY] Trivial Draws/Wins
        if bb.phase < 12:
            # White has only Minor? (B or N) -> Draw
            if not bb.pw[PIECE_P] and not bb.pw[PIECE_Q] and not bb.pw[PIECE_R]:
                if bb.pw[PIECE_B] or bb.pw[PIECE_N]:
                    return 0 # Insufficient material
                    
            # White has only Pawn? (K+P vs K)
            if bb.pw[PIECE_P] and not (bb.pw[PIECE_N] | bb.pw[PIECE_B] | bb.pw[PIECE_R] | bb.pw[PIECE_Q]):
                # Winning Zone Detection
                p_sq = (bb.pw[PIECE_P] & -bb.pw[PIECE_P]).bit_length() - 1
                px, py = p_sq % 5, p_sq // 5
                
                # Distance to promotion (Rank 4)
                dist_promo = 4 - py
                
                if dist_promo <= 2:
                    # Check King Escort
                    if bb.pw[PIECE_K]:
                        ksq = (bb.pw[PIECE_K] & -bb.pw[PIECE_K]).bit_length() - 1
                        kx, ky = ksq % 5, ksq // 5
                        dist_escort = max(abs(px - kx), abs(py - ky))
                        
                        if dist_escort <= 1:
                            # Check Enemy King Distance
                            if bb.pb[PIECE_K]:
                                eksq = (bb.pb[PIECE_K] & -bb.pb[PIECE_K]).bit_length() - 1
                                ekx, eky = eksq % 5, eksq // 5
                                dist_enemy = max(abs(px - ekx), abs(py - eky))
                                
                                # If enemy is too far to stop
                                if dist_enemy > dist_promo + 1:
                                    return 20000
            
    if not w_pieces: # White has only King
        # Black has Major?
        if (bb.pb[PIECE_Q] or bb.pb[PIECE_R]):
            return -20000
            
        # [ENDGAME ONLY] Trivial Draws/Wins
        if bb.phase < 12:
            # Black has only Minor? -> Draw
            if not bb.pb[PIECE_P] and not bb.pb[PIECE_Q] and not bb.pb[PIECE_R]:
                if bb.pb[PIECE_B] or bb.pb[PIECE_N]:
                    return 0
                    
            # Black has only Pawn?
            if bb.pb[PIECE_P] and not (bb.pb[PIECE_N] | bb.pb[PIECE_B] | bb.pb[PIECE_R] | bb.pb[PIECE_Q]):
                p_sq = (bb.pb[PIECE_P] & -bb.pb[PIECE_P]).bit_length() - 1
                px, py = p_sq % 5, p_sq // 5
                
                dist_promo = py # Rank 0
                
                if dist_promo <= 2:
                    if bb.pb[PIECE_K]:
                        ksq = (bb.pb[PIECE_K] & -bb.pb[PIECE_K]).bit_length() - 1
                        kx, ky = ksq % 5, ksq // 5
                        dist_escort = max(abs(px - kx), abs(py - ky))
                        
                        if dist_escort <= 1:
                            if bb.pw[PIECE_K]:
                                eksq = (bb.pw[PIECE_K] & -bb.pw[PIECE_K]).bit_length() - 1
                                ekx, eky = eksq % 5, eksq // 5
                                dist_enemy = max(abs(px - ekx), abs(py - eky))
                                
                                if dist_enemy > dist_promo + 1:
                                    return -20000

    return None

def evaluate(bb: GameState) -> int:
    """
    Simplified, principled evaluation function.
    """
    # 1. Material + PST (Incremental) + Pawn Structure (Cached)
    p_mg, p_eg = eval_pawns(bb)
    
    mg_total = bb.mg + p_mg
    eg_total = bb.eg + p_eg
    
    phase = bb.phase
    if phase > MAX_PHASE: phase = MAX_PHASE
    
    score = (mg_total * phase + eg_total * (MAX_PHASE - phase)) // MAX_PHASE
    
    # 2. Mobility (Weighted)
    w_mob = mobility(bb, WHITE)
    b_mob = mobility(bb, BLACK)
    score += (w_mob - b_mob)
    
    # 3. King Safety (Middlegame Focused)
    w_safe_mg = king_safety(bb, WHITE)
    b_safe_mg = king_safety(bb, BLACK)
    
    # Scale MG safety by Phase (Fade out in endgame)
    score += (w_safe_mg - b_safe_mg) * phase // MAX_PHASE
    
    # Advanced Safety (Universal)
    w_safe_adv = eval_king_mobility(bb, WHITE) + \
                 eval_queen_check_danger(bb, WHITE) + \
                 eval_pawn_mate_support(bb, WHITE)
    
    b_safe_adv = eval_king_mobility(bb, BLACK) + \
                 eval_queen_check_danger(bb, BLACK) + \
                 eval_pawn_mate_support(bb, BLACK)
    
    score += (w_safe_adv - b_safe_adv)
    
    # 4. Strategic & Tactical
    strategic_score = 0
    
    # White
    strategic_score += eval_center_control(bb, WHITE)
    strategic_score += eval_threats(bb, WHITE) 
    strategic_score += eval_fork_potential(bb, WHITE)
    strategic_score += eval_queen_danger(bb, WHITE) 
    strategic_score += eval_coordination(bb, WHITE)
    strategic_score += eval_trapped_right(bb, WHITE) 
    strategic_score += eval_development(bb, WHITE)
    
    # Black
    strategic_score -= eval_center_control(bb, BLACK)
    strategic_score -= eval_threats(bb, BLACK)
    strategic_score -= eval_fork_potential(bb, BLACK)
    strategic_score -= eval_queen_danger(bb, BLACK)
    strategic_score -= eval_coordination(bb, BLACK)
    strategic_score -= eval_trapped_right(bb, BLACK)
    strategic_score -= eval_development(bb, BLACK)
    
    score += strategic_score
    
    # 5. Mop-up
    if score > 500:
        score += eval_mopup(bb, WHITE, score, 0)
    elif score < -500:
        score -= eval_mopup(bb, BLACK, -score, 0)

    # 6. Endgame King Activity & Opposition
    if phase < 12: # Endgame
        w_act = eval_king_activity(bb, WHITE)
        b_act = eval_king_activity(bb, BLACK)
        score += (w_act - b_act)
        
        score += eval_opposition(bb)
        # Apply Endgame Adjustments (if defined, need to check def)
        # score = apply_endgame_adjustments(bb, score) # Re-enable if confirmed existence
        # Piece Endgame Adjustments (removed)
             
    # 7. Trivial Endgames
    trivial = detect_trivial_endgames(bb)
    if trivial is not None:
        return trivial if bb.white_to_move else -trivial

    # 8. Insufficient Material
    if is_insufficient_material(bb):
        return 0

    return score if bb.white_to_move else -score






# =============================================================================
# MVV-LVA / HISTORY / KILLERS / COUNTERMOVE / TT
# =============================================================================

PIECE_ORDER_VAL = [100, 320, 330, 500, 900, 20000]

MVV_LVA = [[0]*7 for _ in range(6)]
for a in range(6):
    for v in range(6):
        MVV_LVA[a][v] = PIECE_ORDER_VAL[v] * 16 - PIECE_ORDER_VAL[a]

HISTORY = [[[0]*SQUARES for _ in range(SQUARES)] for _ in range(6)]
KILLERS = [[0, 0] for _ in range(MAX_PLY)]
COUNTERMOVE = [0] * (SQUARES * SQUARES)

def prev_index(m: Move) -> int:
    return move_from(m) * SQUARES + move_to(m)

# TT entry format:
# (key, depth, score, flag, move)
TT = [None] * TT_SIZE
TT_FLAG_EXACT = 0
TT_FLAG_LOWER = 1
TT_FLAG_UPPER = 2

# =============================================================================
# MOVE ORDERING SCORE
# =============================================================================

# Reverse Futility Pruning margins
RFP_MARGIN = {1: 120, 2: 160, 3: 210, 4: 260, 5: 320}

# =============================================================================
# MOVE ORDERING SCORE
# =============================================================================

def move_order_score(m: Move, ply: int, tt_move: Move, prev_move: Move) -> int:
    if m == tt_move:
        return 1_000_000_000

    score = 0

    if move_is_capture(m):
        a = move_piece(m)
        v = move_captured_piece(m)
        if v > 6:
            v = 6
        score += 800_000 + MVV_LVA[a][v]
    else:
        p = min(ply, MAX_PLY - 1)
        k0, k1 = KILLERS[p]

        if m == k0:
            return 700_000
        if m == k1:
            return 690_000
        
        # Countermove Heuristic
        if prev_move != 0:
            cm = COUNTERMOVE[prev_index(prev_move)]
            if m == cm:
                return 600_000

        p = move_piece(m)
        f = move_from(m)
        t = move_to(m)
        score += HISTORY[p][f][t]
        
        # Continuation History
        # CONT_HISTORY[piece_prev][from_prev][to_prev][to_current]
        # We need prev_move details
        if prev_move != 0:
            pp = move_piece(prev_move)
            pf = move_from(prev_move)
            pt = move_to(prev_move)
            # We use 't' (to_current) as the index? 
            # The table is [6][25][25][25]. Last dim is 25 (SQUARES).
            # So we index by 't'.
            score += CONT_HISTORY[pp][pf][pt][t]

    return score

def eval_king_activity(bb: GameState, color: int) -> int:
    """
    Endgame King Activity:
    1. Centralization (control center).
    2. Advancement (push towards enemy).
    """
    score = 0
    if color == WHITE:
        K = bb.pw[PIECE_K]
    else:
        K = bb.pb[PIECE_K]
        
    if not K: return 0
    
    sq = (K & -K).bit_length() - 1
    x, y = sq_to_xy(sq)
    
    # 1. Centralization (Manhattan distance from 2,2)
    dist = abs(x - 2) + abs(y - 2)
    # Max dist is 2+2=4 (corner). Min is 0.
    # Bonus: (4 - dist) * 5 -> Max 20.
    score += (4 - dist) * 5
    
    # 2. Advancement (Rank)
    # White wants y to be high (4). Black wants y to be low (0).
    if color == WHITE:
        score += y * 10 # Rank 0 -> 0, Rank 4 -> 40
    else:
        score += (4 - y) * 10 # Rank 4 -> 0, Rank 0 -> 40
        
    return score

def eval_opposition(bb: GameState) -> int:
    """
    Returns bonus for White if White has opposition, or penalty if Black has it.
    Generalized Opposition: Kings on same rank/file/diagonal with EVEN distance (odd squares between).
    The side NOT to move has the opposition.
    """
    wk = bb.pw[PIECE_K]
    bk = bb.pb[PIECE_K]
    if not wk or not bk: return 0
    
    wksq = (wk & -wk).bit_length() - 1
    bksq = (bk & -bk).bit_length() - 1
    
    wx, wy = sq_to_xy(wksq)
    bx, by = sq_to_xy(bksq)
    
    dx = abs(wx - bx)
    dy = abs(wy - by)
    
    # Opposition condition: dx, dy are both EVEN (0, 2, 4)
    # And kings are on same line (dx=0, dy=0, or dx=dy)
    is_opposition = False
    if (dx % 2 == 0) and (dy % 2 == 0):
        if dx == 0 or dy == 0 or dx == dy:
            if dx + dy > 0: # Not on same square
                is_opposition = True
        
    if is_opposition:
        # If it is White's turn, Black has opposition -> Penalty for White
        if bb.white_to_move:
            return -30
        else:
            # Black's turn, White has opposition -> Bonus for White
            return 30
            
    return 0

def apply_endgame_adjustments(bb: GameState, score: int) -> int:
    """
    Adjusts FINAL score based on piece configurations.
    1. Bishop vs Knight (Open/Closed)
    2. Opposite Color Bishops (Drawish)
    """
    w_p = bb.pw[PIECE_P].bit_count()
    b_p = bb.pb[PIECE_P].bit_count()
    total_pawns = w_p + b_p
    
    w_n = bb.pw[PIECE_N].bit_count()
    w_b = bb.pw[PIECE_B].bit_count()
    b_n = bb.pb[PIECE_N].bit_count()
    b_b = bb.pb[PIECE_B].bit_count()
    
    # Bishop vs Knight
    # Open position (few pawns) favors Bishop
    if total_pawns < 6:
        if w_b > 0 and w_n == 0 and b_n > 0 and b_b == 0:
            score += 10
        elif b_b > 0 and b_n == 0 and w_n > 0 and w_b == 0:
            score -= 10
            
    # Opposite Color Bishops
    # If 1 Bishop each, no other pieces (except pawns), and different colors
    # Scale score down (drawish)
    w_maj = bb.pw[PIECE_R] | bb.pw[PIECE_Q]
    b_maj = bb.pb[PIECE_R] | bb.pb[PIECE_Q]
    
    if w_b == 1 and b_b == 1 and w_n == 0 and b_n == 0 and not w_maj and not b_maj:
        # Check colors
        wb_sq = (bb.pw[PIECE_B] & -bb.pw[PIECE_B]).bit_length() - 1
        bb_sq = (bb.pb[PIECE_B] & -bb.pb[PIECE_B]).bit_length() - 1
        w_color = (wb_sq % 5 + wb_sq // 5) % 2
        b_color = (bb_sq % 5 + bb_sq // 5) % 2
        
        if w_color != b_color:
            score //= 2 # Halve the advantage
            
    return score

def is_insufficient_material(bb: GameState) -> bool:
    """
    Returns True if the material on board is insufficient to force a mate.
    """
    # If there are pawns, it's not insufficient (usually)
    if bb.pw[PIECE_P] or bb.pb[PIECE_P]:
        return False
        
    # If there are Queens or Rooks (Right pieces), usually sufficient
    if bb.pw[PIECE_Q] or bb.pb[PIECE_Q] or bb.pw[PIECE_R] or bb.pb[PIECE_R]:
        return False
        
    # Count minors
    w_n = bb.pw[PIECE_N].bit_count()
    w_b = bb.pw[PIECE_B].bit_count()
    b_n = bb.pb[PIECE_N].bit_count()
    b_b = bb.pb[PIECE_B].bit_count()
    
    w_minors = w_n + w_b
    b_minors = b_n + b_b
    
    # K vs K
    if w_minors == 0 and b_minors == 0:
        return True
        
    # K+N vs K or K+B vs K
    if (w_minors == 1 and b_minors == 0) or (w_minors == 0 and b_minors == 1):
        return True
        
    # K+N vs K+N (Draw in 5x5?) - Likely yes
    if w_minors == 1 and b_minors == 1:
        return True
        
    return False

# =============================================================================
# QUIESCENCE SEARCH (capture-only)
# =============================================================================

def _cap_order(m: Move) -> int:
    a = move_piece(m)
    v = move_captured_piece(m)
    if v > 6:
        v = 6
    return MVV_LVA[a][v]

QS_MAX_DEPTH = 12   # safety cap (reduced from 32)

def see_winning(bb: GameState, m: Move) -> bool:
    """Simple SEE: return True if capture is non-losing."""
    attacker = move_piece(m)             # already piece index 0–6
    t = move_to(m)                       # destination square
    # if we take a queen with a pawn → definitely good
    # CAPTURED PIECE TYPE MUST BE LOOKED UP FROM BOARD
    to_sq = move_to(m)
    victim = get_captured_type(bb, WHITE if bb.occ_w & (1 << to_sq) else BLACK, to_sq)

    if victim == CAPTURE_NONE:
        return True

    return PIECE_ORDER_VAL[victim] >= PIECE_ORDER_VAL[attacker]

def capture_order_score(bb: GameState, m: Move) -> int:
    """Calculate move ordering score for captures."""
    if not move_is_capture(m):
        return -10_000_000

    attacker = move_piece(m)
    victim = move_captured_piece(m)
    if victim > 6:
        victim = 6

    base = MVV_LVA[attacker][victim]

    see_val = static_exchange_eval(bb, m)
    if see_val >= 0:
        base += 8_000 + see_val
    else:
        base -= 8_000 - see_val # Penalize bad captures

    return base

def qsearch(bb: GameState, wtm: bool,
            alpha: int, beta: int,
            ply: int,
            start: float, hard: float,
            qdepth: int = 0,
            prev_move: Move = 0) -> Tuple[int, int]:
    """
    Quiescence search to settle dynamic positions.
    """

    if time.perf_counter() >= hard:
        raise TimeoutError

    if qdepth >= QS_MAX_DEPTH:
        return evaluate(bb), 1

    color = WHITE if wtm else BLACK
    # Stand-pat (only if not in check)
    in_chk = in_check(bb, color)
    stand = -MATE_SCORE
    if not in_chk:
        stand = evaluate(bb)
        if stand >= beta:
            return stand, 1
        if stand > alpha:
            alpha = stand

    # ------------------ Move generation
    moves = generate_moves(bb, color)

    # Filter moves
    # If not in check, search captures AND checks AND promotions
    # [ENDGAME ONLY] Also search important quiet moves if phase < 12
    if not in_chk:
        interesting_moves = []
        for m in moves:
            is_cap = move_is_capture(m)
            is_promo = move_is_promotion(m)
            
            # Check if gives check (optimized)
            gives_chk = False
            if not is_cap and not is_promo:
                if gives_check_fast(bb, m):
                    gives_chk = True
            
            is_recapture = False
            if prev_move != 0 and move_to(m) == move_to(prev_move):
                is_recapture = True
                
            is_quiet_important = False
            # [ENDGAME ONLY] Strict Quiet Moves
            if bb.phase < 12 and not is_cap and not is_promo and not gives_chk:
                # 1. Pawn pushes near promotion
                if move_piece(m) == PIECE_P:
                    to_sq = move_to(m)
                    y = to_sq // 5
                    if color == WHITE:
                        if y >= 3: is_quiet_important = True
                    else:
                        if y <= 1: is_quiet_important = True
                        
                # 2. King moves
                if move_piece(m) == PIECE_K:
                    is_quiet_important = True
                    
                # 3. Stop Promotion
                to_sq = move_to(m)
                tx, ty = to_sq % 5, to_sq // 5
                if color == WHITE:
                    if ty < 4 and (bb.pb[PIECE_P] & (1 << ((ty+1)*5 + tx))):
                        is_quiet_important = True
                else:
                    if ty > 0 and (bb.pw[PIECE_P] & (1 << ((ty-1)*5 + tx))):
                        is_quiet_important = True

            if is_cap or is_promo or gives_chk or is_recapture or is_quiet_important:
                interesting_moves.append(m)
                
        moves = interesting_moves
        if not moves:
            return alpha, 1

    # If in check, must search all moves (already in moves)
    if in_chk:
        # Limit check depth in qsearch to avoid explosion
        if qdepth > 2: 
            return evaluate(bb), 1

    # Sorting
    def q_score(m):
        if move_is_capture(m):
            return _cap_order(m) + 1000
        return 0 # Checks/Quiets sorted later? Or just 0.

    moves.sort(key=q_score, reverse=True)

    nodes = 0
    has_legal = False
    
    for m in moves:
        if time.perf_counter() >= hard:
            raise TimeoutError

        is_cap = move_is_capture(m)
        is_promo = move_is_promotion(m)
        
        # Delta pruning (white POV)
        if not in_chk and is_cap and not is_promo:
            # Check gives check
            gives_chk = False
            if gives_check_fast(bb, m):
                gives_chk = True
            
            if not gives_chk:
                victim = move_captured_piece(m)
                if victim != CAPTURE_NONE:
                    if stand + PIECE_ORDER_VAL[victim] + 200 <= alpha:
                        continue

                # SEE pruning (white POV)
                if static_exchange_eval(bb, m) < 0:
                    continue

        undo, next_wtm = step_forward(bb, m, wtm)

        # Illegal move check
        if in_check(bb, color):
            step_back(bb, undo)
            continue
            
        has_legal = True
        
        try:
            child_score, n = qsearch(
                bb, next_wtm,
                -beta, -alpha,     # NEGAMAX WINDOW
                ply + 1,
                start, hard,
                qdepth + 1,
                m # Pass as prev_move
            )
            child_score = -child_score   # NEGAMAX SCORE FLIP
            nodes += n
        except TimeoutError:
            step_back(bb, undo)
            raise
            
        step_back(bb, undo)

        if child_score >= beta:
            return child_score, nodes
        if child_score > alpha:
            alpha = child_score

    if in_chk and not has_legal:
        return -MATE_SCORE + ply // 2, nodes

    return alpha, nodes



# =============================================================================
# STATIC EXCHANGE EVALUATION (SEE)
# =============================================================================

def get_lowest_attacker(bb: GameState, sq: int, color: int,
                       occ: int,
                       my_p: int, my_n: int, my_b: int, my_r: int, my_q: int, my_k: int) -> Tuple[int, int]:
    """
    Finds the least valuable attacker for 'color' on 'sq'.
    Returns (attacker_sq, piece_value).
    """
    x, y = sq % 5, sq // 5
    
    if color == WHITE:
        # Check for white pawns at (x-1, y+1) and (x+1, y+1)
        if y < 4:
            if x > 0:
                from_sq = sq + 5 - 1
                if (my_p & (1 << from_sq)): return from_sq, 120
            if x < 4:
                from_sq = sq + 5 + 1
                if (my_p & (1 << from_sq)): return from_sq, 120
    else:
        # Check for black pawns at (x-1, y-1) and (x+1, y-1)
        if y > 0:
            if x > 0:
                from_sq = sq - 5 - 1
                if (my_p & (1 << from_sq)): return from_sq, 120
            if x < 4:
                from_sq = sq - 5 + 1
                if (my_p & (1 << from_sq)): return from_sq, 120

    # 2. Knight
    knights = my_n
    if knights:
        att = PRE_KNIGHT[sq]
        for from_sq in att:
            if knights & (1 << from_sq):
                return from_sq, 360

    # 3. Bishop
    bishops = my_b
    if bishops:
        att = bishop_attacks_from(sq, occ) & bishops
        if att:
            return (att & -att).bit_length() - 1, 340

    # 4. Rook (and Right piece rook-part)
    rooks = my_r
    if rooks:
        att = rook_attacks_from(sq, occ) & rooks
        if att:
            return (att & -att).bit_length() - 1, 850
            
    # Right piece (Knight part)
    if rooks:
        att = PRE_KNIGHT[sq]
        for from_sq in att:
            if rooks & (1 << from_sq):
                return from_sq, 850

    # 5. Queen
    queens = my_q
    if queens:
        att = (bishop_attacks_from(sq, occ) | rook_attacks_from(sq, occ)) & queens
        if att:
            return (att & -att).bit_length() - 1, 900

    # 6. King
    kings = my_k
    if kings:
        att = PRE_KING[sq]
        for from_sq in att:
            if kings & (1 << from_sq):
                return from_sq, 20000

    return -1, 0

def static_exchange_eval(bb: GameState, move: int) -> int:
    """
    Returns the static exchange evaluation (SEE) score for a move.
    Positive = good capture, Negative = bad capture.
    """
    to_sq = move_to(move)
    from_sq = move_from(move)
    
    cap_type = move_captured_piece(move)
    if cap_type == CAPTURE_NONE:
        return 0
        
    vals = [120, 360, 340, 850, 900, 20000]
    
    gain = [0] * 32
    d = 0
    
    gain[d] = vals[cap_type]
    
    src_piece = move_piece(move)
    occ = bb.occ_all
    occ &= ~(1 << from_sq)
    
    attacker_val = vals[src_piece]
    stm = BLACK if (bb.occ_w & (1 << from_sq)) else WHITE
    
    while True:
        d += 1
        gain[d] = attacker_val - gain[d-1]
        
        if max(-gain[d-1], gain[d]) < 0:
            break
            
        if stm == WHITE:
            my_p = bb.pw[PIECE_P] & occ
            my_n = bb.pw[PIECE_N] & occ
            my_b = bb.pw[PIECE_B] & occ
            my_r = bb.pw[PIECE_R] & occ
            my_q = bb.pw[PIECE_Q] & occ
            my_k = bb.pw[PIECE_K] & occ
        else:
            my_p = bb.pb[PIECE_P] & occ
            my_n = bb.pb[PIECE_N] & occ
            my_b = bb.pb[PIECE_B] & occ
            my_r = bb.pb[PIECE_R] & occ
            my_q = bb.pb[PIECE_Q] & occ
            my_k = bb.pb[PIECE_K] & occ
            
        att_sq, att_val = get_lowest_attacker(bb, to_sq, stm, occ, my_p, my_n, my_b, my_r, my_q, my_k)
        
        if att_sq == -1:
            break
            
        occ &= ~(1 << att_sq)
        attacker_val = att_val
        stm = 1 - stm
        
    while d > 1:
        d -= 1
        gain[d-1] = -max(-gain[d-1], gain[d])
        
    return gain[0]

# =============================================================================
# QUIESCENCE SEARCH
# =============================================================================



def negamax(bb: GameState, white_to_move: bool,
            depth: int, alpha: int, beta: int,
            ply: int, prev_move: Move,
            start_time: float, soft_time: float, hard_time: float,
            excluded_move: Move = 0) -> Tuple[int, int]:
    """
    Principal Variation Search (PVS) with Negamax.
    """

    # with open("debug.log", "a") as f:
    #     f.write(f"negamax depth={depth} ply={ply}\n")

    if time.perf_counter() >= hard_time:
        raise TimeoutError
    if ply >= MAX_PLY:
        return evaluate(bb), 1

    # -------------------------- 0. REPETITION CHECK
    # Check if we have seen this position before in the current path
    if bb.zkey in bb.history:
        return 0, 1

    nodes = 0
    color = WHITE if white_to_move else BLACK
    orig_alpha = alpha

    # -------------------------- 1. TT PROBE
    key = bb.zkey
    tti = key & TT_MASK
    entry = TT[tti]

    tt_move = 0
    if entry is not None and entry[0] == key:
        stored_depth = entry[1]
        stored_score = entry[2]
        
        # Adjust mate scores from TT (relative to current ply)
        if abs(stored_score) > MATE_SCORE - 100:
            if stored_score > 0:
                stored_score -= ply // 2
            else:
                stored_score += ply // 2

        stored_flag  = entry[3]
        tt_move      = entry[4]

        # Only use TT if we are NOT excluding a move (standard search)
        # If we are in a singular search (excluded_move != 0), we can't trust the TT score 
        # because it might come from the move we are excluding.
        if excluded_move == 0:
            if stored_depth >= depth:
                if stored_flag == TT_FLAG_EXACT:
                    return stored_score, 1
                if stored_flag == TT_FLAG_LOWER and stored_score > alpha:
                    alpha = stored_score
                elif stored_flag == TT_FLAG_UPPER and stored_score < beta:
                    beta = stored_score
                if alpha >= beta:
                    return stored_score, 1

    # -------------------------- 1.5 INTERNAL ITERATIVE DEEPENING (IID)
    if depth >= 4 and tt_move == 0:
        # Search shallower to find a likely best move
        try:
            negamax(bb, white_to_move, depth - 2, alpha, beta, ply, prev_move, start_time, soft_time, hard_time)
        except TimeoutError:
            pass
        # Re-probe TT
        entry = TT[tti]
        if entry is not None and entry[0] == key:
            tt_move = entry[4]

    # -------------------------- 2. LEAF → QSEARCH
    if depth <= 0:
        return qsearch(bb, white_to_move, alpha, beta, ply, start_time, hard_time, prev_move=prev_move)

    # -------------------------- 3. CHECK / RAZOR / FUTILITY / RFP
    in_chk = in_check(bb, color)
    
    # Extensions (Check & Mate Threat)
    # Ensure these run before pruning decisions
    if in_chk:
        # Limit check extensions to +1 ply max per branch
        if ply < 20: depth += 1

    # Static Eval
    if not in_chk:
        stand = evaluate(bb)
    else:
        stand = -MATE_SCORE

    # Null Move Pruning (NMP)
    # [ENDGAME ONLY] Strict conditions
    is_safe = not in_chk
    if depth >= 3 and stand >= beta and is_safe:
        can_null = True
        
        # [ENDGAME ONLY] Strict NMP
        if bb.phase < 12:
            if abs(stand) < 100: can_null = False # Unstable eval
            if abs(stand) >= 9000: can_null = False # Don't prune mate scores
            
            # Check pawns
            if white_to_move:
                if not bb.pw[PIECE_P]: can_null = False
            else:
                if not bb.pb[PIECE_P]: can_null = False
                
            # Check King Mobility
            if eval_king_mobility(bb, color) <= -50: can_null = False
            
        if can_null:
            # Verify we have pieces (standard NMP check)
            has_pieces = (bb.pw[PIECE_N] | bb.pw[PIECE_B] | bb.pw[PIECE_R] | bb.pw[PIECE_Q]) if white_to_move else \
                         (bb.pb[PIECE_N] | bb.pb[PIECE_B] | bb.pb[PIECE_R] | bb.pb[PIECE_Q])
            
            if has_pieces:
                R = 1 # Reduced from 2/3 for 5x5
                
                # Make null move (toggle side, no board changes)
                bb.zkey ^= Z_SIDE
                
                try:
                    score, n = negamax(bb, not white_to_move, depth - R - 1, -beta, -beta + 1, ply + 1, 0, start_time, soft_time, hard_time)
                    score = -score
                except TimeoutError:
                    bb.zkey ^= Z_SIDE # Undo
                    raise
                
                bb.zkey ^= Z_SIDE # Undo
                
                if score >= beta:
                    return beta, 1

    # -------------------------- 3.6 SINGULAR EXTENSIONS
    # If the TT move is "singular" (much better than alternatives), extend depth.
    # Only do this if we have a strong TT move and enough depth.
    extension = 0
    if (depth >= 8 and tt_move != 0 and excluded_move == 0 and 
        entry is not None and 
        abs(entry[2]) < MATE_SCORE and 
        entry[1] >= depth - 3 and # TT depth must be reasonable
        (entry[3] == TT_FLAG_EXACT or entry[3] == TT_FLAG_LOWER)):
        
        # Margin: how much worse can other moves be?
        # If best_move_score - margin > other_moves_best_score, then it's singular.
        # We test if any other move can beat (tt_score - margin).
        se_margin = 2 * depth
        singular_beta = entry[2] - se_margin
        
        # Search with reduced depth, excluding the TT move
        # We want to see if we fail low (return <= singular_beta).
        # If we fail low, it means no other move could beat singular_beta.
        try:
            val, _ = negamax(bb, white_to_move, depth - 3, singular_beta - 1, singular_beta, 
                             ply, prev_move, start_time, soft_time, hard_time, 
                             excluded_move=tt_move)
            
            if val < singular_beta:
                extension = 1
        except TimeoutError:
            pass

    # -------------------------- 4. MOVE GENERATION
    moves = generate_moves(bb, color)
    
    # Move Ordering
    # ... (Standard ordering logic) ...
    def score_move(m):
        if m == tt_move:
            return 2000000
        
        # Captures first
        if (m >> 10) & 1:
            return 100000 + _cap_order(m)

        # Promotions next
        if (m >> 18) & 1:
            return 95000

        # Then checks
        if gives_check_fast(bb, m):
            return 90000

        # Quiet: history heuristic
        fr = (m >> 5) & 0x1F
        to = m & 0x1F
        p = (m >> 11) & 7
        return HISTORY[p][fr][to]


    moves.sort(key=score_move, reverse=True)
    
    # -------------------------- 5. LOOP
    best_score = -MATE_SCORE
    best_move = 0
    
    # Futility Pruning (5x5 Tuned)
    # [ENDGAME ONLY] Strict conditions
    if depth <= 6 and not in_chk:
        futility_margin = 45 * depth
        
        is_promo_threat = False
        # Check for promotion threats (Piece near promotion)
        if color == WHITE:
            if (bb.pw[PIECE_P] & RANK_MASKS[3]) or (bb.pw[PIECE_P] & RANK_MASKS[4]): is_promo_threat = True
        else:
            if (bb.pb[PIECE_P] & RANK_MASKS[1]) or (bb.pb[PIECE_P] & RANK_MASKS[0]): is_promo_threat = True
            
        if not is_promo_threat:
            # Check Hanging Pieces (Expensive) - ONLY IN ENDGAME
            has_hanging = False
            if bb.phase < 12:
                has_hanging = eval_threats(bb, color) < -20
            
            # Don't prune mate scores
            is_mate = abs(stand) >= 9000
            
            if not has_hanging and not is_mate:
                if stand + futility_margin <= alpha:
                    # Prune if not a capture/promotion/check
                    # We check inside loop
                    pass

    has_legal = False
    
    for idx, m in enumerate(moves):
        is_important = True
        
        # Futility Pruning Check inside loop
        if depth <= 6 and not in_chk and not move_is_capture(m) and not move_is_promotion(m):
            # Re-check conditions that depend on move? No, already checked static conditions.
            # Just check if we decided to prune.
            # But we need to verify the move doesn't give check.
            if not is_promo_threat and not has_hanging and not is_mate:
                 if stand + futility_margin <= alpha:
                    # Check if gives check
                    undo_f, next_wtm_f = step_forward(bb, m, white_to_move)
                    gives_check = in_check(bb, WHITE if next_wtm_f else BLACK)
                    step_back(bb, undo_f)
                    
                    if not gives_check:
                        continue

        # Late Move Pruning (LMP)
        # Only prune moveIndex > 12
        if depth <= 4 and idx > 12 and not in_chk and not move_is_capture(m) and not move_is_promotion(m):
            # [NEW] Don't prune mate scores
            if abs(stand) >= 9000:
                pass
            else:
                is_important = False
                
                # 1. Checking moves (Always important)
                undo_l, next_wtm_l = step_forward(bb, m, white_to_move)
                if in_check(bb, WHITE if next_wtm_l else BLACK):
                    is_important = True
                step_back(bb, undo_l)
                
                # [ENDGAME ONLY] Strict Exceptions
                if not is_important and bb.phase < 12:
                    p_type = move_piece(m)
                    if p_type == PIECE_K: is_important = True
                    if p_type == PIECE_P: is_important = True
                    
                    if not is_important:
                        # Check if touches King mobility
                        k_sq = (bb.pw[PIECE_K] if white_to_move else bb.pb[PIECE_K]).bit_length() - 1
                        to_sq = move_to(m)
                        kx, ky = sq_to_xy(k_sq)
                        tx, ty = sq_to_xy(to_sq)
                        if max(abs(kx - tx), abs(ky - ty)) <= 1:
                            is_important = True
                
                if not is_important:
                    continue

        undo, next_wtm = step_forward(bb, m, white_to_move)

        if in_check(bb, color):
            step_back(bb, undo)
            continue
            
        has_legal = True
        
        # PVS
        score = 0
        if idx == 0:
            try:
                score, n = negamax(bb, next_wtm, depth - 1, -beta, -alpha, ply + 1, m, start_time, soft_time, hard_time)
                score = -score
                nodes += n
            except TimeoutError:
                step_back(bb, undo)
                raise
        else:
            # LMR (Beginner Engine Formula)
            reduction = 0
            if (not in_chk and not move_is_capture(m) and not move_is_promotion(m) 
                and idx > 3 and depth >= 3):
                reduction = 1
            
            try:
                # First search: Reduced depth (if applicable), Zero Window
                score, n = negamax(bb, next_wtm, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, m, start_time, soft_time, hard_time)
                score = -score
                nodes += n
            except TimeoutError:
                step_back(bb, undo)
                raise
            
            # Re-search if LMR worked too well (we reduced, but it beat alpha)
            if reduction > 0 and score > alpha:
                 try:
                    score, n = negamax(bb, next_wtm, depth - 1, -alpha - 1, -alpha, ply + 1, m, start_time, soft_time, hard_time)
                    score = -score
                    nodes += n
                 except TimeoutError:
                    step_back(bb, undo)
                    raise
            
            # PVS Re-search (Full Window) if move is better than alpha but within bounds
            if alpha < score < beta:
                try:
                    score, n = negamax(bb, next_wtm, depth - 1, -beta, -alpha, ply + 1, m, start_time, soft_time, hard_time)
                    score = -score
                    nodes += n
                except TimeoutError:
                    step_back(bb, undo)
                    raise

        
        step_back(bb, undo)
        
        if score > best_score:
            best_score = score
            best_move = m
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break # Beta Cutoff

    # -------------------------- 7. TT STORE
    if not has_legal:
        # No legal moves found in the loop (Checkmate or Stalemate)
        # In this variant, Stalemate is a LOSS.
        return -MATE_SCORE + ply // 2, 1

    flag = TT_FLAG_EXACT
    if best_score <= orig_alpha:
        flag = TT_FLAG_UPPER
    elif best_score >= beta:
        flag = TT_FLAG_LOWER

    # Adjust mate scores for TT storage (relative to ply)
    tt_score = best_score
    if abs(best_score) > MATE_SCORE - 100:
        if best_score > 0:
            tt_score += ply // 2
        else:
            tt_score -= ply // 2

    if entry is None or depth >= entry[1]:
        TT[tti] = (key, depth, tt_score, flag, best_move)

    return best_score, nodes




# =============================================================================
# ROOT SEARCH
# =============================================================================
def search_root(bb: GameState, white_to_move: bool,
                max_depth: int, time_limit: float) -> Tuple[int, Move, int]:
    """
    Root entry point for Iterative Deepening search.
    """

    start = time.perf_counter()
    # Use the passed time_limit (which is already min(budget, SAFE_HARD))
    HARD = start + time_limit
    # Soft limit as DURATION
    SOFT_LIMIT = time_limit * 0.95

    color = WHITE if white_to_move else BLACK
    moves = generate_moves(bb, color)

    # --- FILTER ILLEGAL MOVES ---
    legal_moves = []
    for m in moves:
        undo, _ = step_forward(bb, m, white_to_move)
        if not in_check(bb, color):
            legal_moves.append(m)
        step_back(bb, undo)
    moves = legal_moves

    if not moves:
        # No legal moves at root = Loss (Checkmate or Stalemate)
        return -MATE_SCORE, 0, 1

    # Optimization: If only one legal move, we still search it to get a score (e.g. mate)
    if len(moves) == 1:
        # print(f"[SEARCH] Single legal move: {str(moves[0])}")
        best_move = moves[0]
        best_score = 0
        total_nodes = 0
        
        # Run ID loop just for this move to get accurate score/mate
        for depth in range(1, max_depth + 1):
            if time.perf_counter() >= HARD:
                break
                
            undo, next_wtm = step_forward(bb, best_move, white_to_move)
            try:
                # Full window search on the single move
                score, n = negamax(bb, next_wtm, depth - 1, -1_000_000_000, 1_000_000_000, 1, best_move, start, SOFT_LIMIT, HARD)
                score = -score
                total_nodes += n
                best_score = score
            except TimeoutError:
                step_back(bb, undo)
                break
            step_back(bb, undo)
            
            # Format score
            score_str = f"cp {best_score}"
            if abs(best_score) > MATE_SCORE - 100:
                if best_score > 0:
                    mate_in = (MATE_SCORE - best_score + 1) // 2
                    score_str = f"mate {mate_in}"
                else:
                    mate_in = (-MATE_SCORE - best_score) // 2
                    score_str = f"mate {mate_in}"
            
            # print(f"info depth {depth} score {score_str} nodes {total_nodes} pv {best_move}")
            
            # Stop if mate found
            if abs(best_score) > MATE_SCORE - 100:
                break
                
        return best_score, best_move, total_nodes

    best_move_prev = moves[0]
    best_score_prev = 0
    total_nodes = 0

    for depth in range(1, max_depth + 1):
        # print(f"[SEARCH] Starting depth {depth}")

        # Check if we found a forced mate in the previous iteration
        if abs(best_score_prev) > MATE_SCORE - 100:
            # print(f"[SEARCH] Mate found at depth {depth-1}, stopping.")
            break
            
        alpha = -1_000_000_000
        beta = 1_000_000_000
        
        best_score = -1_000_000_000
        best_move = 0
        
        # Sort moves: PV first
        if depth > 1:
            moves.sort(key=lambda m: 1 if m == best_move_prev else 0, reverse=True)
            
        for idx, m in enumerate(moves):
            if time.perf_counter() >= HARD:
                break
                
            undo, next_wtm = step_forward(bb, m, white_to_move)
            
            try:
                # PVS at root
                if idx == 0:
                    score, n = negamax(bb, next_wtm, depth - 1, -beta, -alpha, 1, m, start, SOFT_LIMIT, HARD)
                else:
                    score, n = negamax(bb, next_wtm, depth - 1, -alpha - 1, -alpha, 1, m, start, SOFT_LIMIT, HARD)
                    if alpha < -score < beta:
                        score, n = negamax(bb, next_wtm, depth - 1, -beta, -alpha, 1, m, start, SOFT_LIMIT, HARD)
                        
                score = -score
                total_nodes += n
            except TimeoutError:
                step_back(bb, undo)
                break
                
            step_back(bb, undo)
            
            if score > best_score:
                best_score = score
                best_move = m
                
            if score > alpha:
                alpha = score
                
        if time.perf_counter() >= HARD:
            break
            
        best_score_prev = best_score
        best_move_prev = best_move
        
        # Format score
        score_str = f"cp {best_score}"
        if abs(best_score) > MATE_SCORE - 100:
            if best_score > 0:
                mate_in = (MATE_SCORE - best_score + 1) // 2
                score_str = f"mate {mate_in}"
            else:
                mate_in = (-MATE_SCORE - best_score) // 2
                score_str = f"mate {mate_in}"
        
        # print(f"info depth {depth} score {score_str} nodes {total_nodes} pv {best_move}")

    return best_score_prev, best_move_prev, total_nodes




    return best_score_prev, best_move_prev, total_nodes



# =============================================================================
# INITIALIZATION & CONVERSION
# =============================================================================

def eval_init_compute(pieces: List[List[int]]) -> Tuple[int, int, int]:
    """
    Replaces compute_initial_eval().
    Computes mg/eg/phase but returns integers directly.
    """
    mg = 0
    eg = 0
    phase = 0

    # white pieces
    for p in range(6):
        b = pieces[0][p]
        while b:
            lsb = b & -b
            sq = lsb.bit_length() - 1
            b &= b - 1
            mg += PIECE_VALUE_MG[p] + PST_MG[WHITE][p][sq]
            eg += PIECE_VALUE_EG[p] + PST_EG[WHITE][p][sq]
            phase += PHASE_WEIGHTS[p]

    # black pieces
    for p in range(6):
        b = pieces[1][p]
        while b:
            lsb = b & -b
            sq = lsb.bit_length() - 1
            b &= b - 1
            mg -= PIECE_VALUE_MG[p] + PST_MG[BLACK][p][sq]
            eg -= PIECE_VALUE_EG[p] + PST_EG[BLACK][p][sq]
            phase += PHASE_WEIGHTS[p]
    return mg, eg, phase

def _compute_zkey_initial(WP: int, WN: int, WB: int, WR: int, WQ: int, WK: int,
                          BP: int, BN: int, BB: int, BR: int, BQ: int, BK: int,
                          white_to_move: bool) -> int:
    """Compute the initial Zobrist key from piece bitboards."""
    h = 0
    # white
    for p, bm in enumerate((WP,WN,WB,WR,WQ,WK)):
        b = bm
        while b:
            lsb = b & -b
            sq = lsb.bit_length() - 1
            h ^= Z_PIECE[WHITE][p][sq]
            b &= b - 1
    # black
    for p, bm in enumerate((BP,BN,BB,BR,BQ,BK)):
        b = bm
        while b:
            lsb = b & -b
            sq = lsb.bit_length() - 1
            h ^= Z_PIECE[BLACK][p][sq]
            b &= b - 1
    if white_to_move:
        h ^= Z_SIDE
    return h

def board_to_corestate5x5(board: Any, white_to_move: bool) -> GameState:
    """Convert platform board object to internal GameState."""
    # 12 pure-integer bitboards
    WP = WN = WB = WR = WQ = WK = 0
    BP = BN = BB = BR = BQ = BK = 0

    # Fill bitboards
    for piece in board.get_pieces():
        sq = xy_to_sq(piece.position.x, piece.position.y)
        mask = 1 << sq

        c = WHITE if piece.player.name == "white" else BLACK
        
        # Robust fallback: use string representation
        # piece.name is behaving inconsistently (method vs property)
        s = str(piece).upper()
        if "KNIGHT" in s:
            pchar = "N"
        else:
            pchar = s[0]
            
        if pchar not in PIECE_FROM_CHAR:
            continue
        p = PIECE_FROM_CHAR[pchar]

        if c == WHITE:
            if p == PIECE_P: WP |= mask
            elif p == PIECE_N: WN |= mask
            elif p == PIECE_B: WB |= mask
            elif p == PIECE_R: WR |= mask
            elif p == PIECE_Q: WQ |= mask
            elif p == PIECE_K: WK |= mask
        else:
            if p == PIECE_P: BP |= mask
            elif p == PIECE_N: BN |= mask
            elif p == PIECE_B: BB |= mask
            elif p == PIECE_R: BR |= mask
            elif p == PIECE_Q: BQ |= mask
            elif p == PIECE_K: BK |= mask

    # Occupancy
    occ_w = WP | WN | WB | WR | WQ | WK
    occ_b = BP | BN | BB | BR | BQ | BK
    occ_all = occ_w | occ_b

    # Initial eval (MG, EG, PHASE)
    pieces = [
        [WP,WN,WB,WR,WQ,WK],
        [BP,BN,BB,BR,BQ,BK],
    ]
    mg, eg, phase = eval_init_compute(pieces)

    # Initial zobrist
    zkey = _compute_zkey_initial(
        WP,WN,WB,WR,WQ,WK,
        BP,BN,BB,BR,BQ,BK,
        white_to_move
    )

    bb = GameState(
        WP,WN,WB,WR,WQ,WK,
        BP,BN,BB,BR,BQ,BK,
        occ_w,occ_b,occ_all,
        mg,eg,phase,zkey,white_to_move
    )
    bb.mob_w = mobility_side(WHITE, bb)
    bb.mob_b = mobility_side(BLACK, bb)
    return bb

# =============================================================================
# MAP ENGINE MOVE → PLATFORM MOVE
# =============================================================================

def _extract_move_destination(move_obj: Any) -> Optional[Tuple[int, int]]:
    """Extract (x, y) destination from various platform move object formats."""
    for attr in ("to", "target", "dest", "destination", "position"):
        if hasattr(move_obj, attr):
            pos = getattr(move_obj, attr)
            if hasattr(pos, "x") and hasattr(pos, "y"):
                return pos.x, pos.y

    if hasattr(move_obj, "x") and hasattr(move_obj, "y"):
        return move_obj.x, move_obj.y

    if isinstance(move_obj, (tuple, list)):
        if len(move_obj) == 2 and all(isinstance(v, int) for v in move_obj):
            return move_obj[0], move_obj[1]
        if len(move_obj) == 2 and isinstance(move_obj[1], (tuple,list)):
            return move_obj[1][0], move_obj[1][1]

    return None

def map_move_to_platform(board: Any, player: Any, m: Move) -> Tuple[Any, Any]:
    """Map internal engine move to platform move object."""
    fr = move_from(m)
    to = move_to(m)
    fx, fy = sq_to_xy(fr)
    tx, ty = sq_to_xy(to)

    legal = list_legal_moves_for(board, player)

    for piece_obj, move_obj in legal:
        if piece_obj.position.x != fx: continue
        if piece_obj.position.y != fy: continue

        dest = _extract_move_destination(move_obj)
        if dest is None:
            continue

        dx, dy = dest
        if dx == tx and dy == ty:
            return piece_obj, move_obj

    return None, None

# =============================================================================
# AGENT ENTRYPOINT
# =============================================================================

def agent(board: Any, player: Any, var: Tuple[int, float]) -> Tuple[Any, Any]:
    """
    Main agent entry point.
    var: (ply_id, time_budget)
    """
    ply_id, time_budget = var

    SAFE_TIME = min(time_budget, SAFE_HARD)

    tstart = time.perf_counter()
    pname = getattr(player, "name", "white")
    if callable(pname):
        pname = pname()
    white_to_move = (str(pname).lower() == "white")

    bb = board_to_corestate5x5(board, white_to_move)

    score, best_move, nodes = search_root(
        bb,
        white_to_move,
        MAX_DEPTH_CAP,
        SAFE_TIME
    )

    piece_obj = None
    move_obj = None

    if best_move:
        try:
            piece_obj, move_obj = map_move_to_platform(board, player, best_move)
        except Exception:
            piece_obj, move_obj = None, None

    if not piece_obj or not move_obj:
        legal = list_legal_moves_for(board, player)
        if legal:
            piece_obj, move_obj = random.choice(legal)

    elapsed = time.perf_counter() - tstart

    return piece_obj, move_obj
