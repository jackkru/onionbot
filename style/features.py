"""
Board position → neural network input tensor.

Encoding: 18 binary planes of shape (8, 8).

Planes 0-5:   White pieces  [P, N, B, R, Q, K]
Planes 6-11:  Black pieces  [P, N, B, R, Q, K]
Plane 12:     White can castle kingside  (filled with 1.0 or 0.0)
Plane 13:     White can castle queenside
Plane 14:     Black can castle kingside
Plane 15:     Black can castle queenside
Plane 16:     Side to move  (1.0 = white to move, 0.0 = black to move)
Plane 17:     En passant target square  (1.0 at the ep square, 0.0 elsewhere)

Moves are encoded as a flat integer: from_square * 64 + to_square.
This gives a vocabulary of 4096 move indices.
Promotions always use the queen (underpromotions are rare and handled gracefully).
"""

from typing import List

import numpy as np
import chess

NUM_PLANES = 18
MOVE_VOCAB = 64 * 64  # 4096

_PIECE_TO_PLANE = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board to a float32 array of shape (18, 8, 8).

    Returns:
        numpy array of shape (NUM_PLANES, 8, 8), dtype float32.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        plane = _PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane += 6
        planes[plane, rank, file] = 1.0

    planes[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16, :, :] = float(board.turn == chess.WHITE)

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        planes[17, rank, file] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    """
    Encode a move as a flat integer in [0, 4095].
    Promotions are collapsed to their from/to squares (queen assumed).
    """
    return move.from_square * 64 + move.to_square


def legal_move_indices(board: chess.Board) -> List[int]:
    """Return a deduplicated list of move indices for all legal moves."""
    seen = set()
    indices = []
    for move in board.legal_moves:
        idx = move_to_index(move)
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices
