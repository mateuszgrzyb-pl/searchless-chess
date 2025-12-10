"""Module with utility functions for file processing."""
from typing import Optional

import chess
import numpy as np
import pandas as pd


PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def allowed_pieces_only(fen: str, allowed_pieces: str) -> bool:
    """Checks if only allowed pieces are present in the given FEN string.

    Args:
        fen: FEN notation string (full format). Only the piece placement section
            (first field before space) is validated.
        allowed_pieces: String containing allowed piece symbols. Case-insensitive.

    Returns:
        True if all letters in the FEN placement section belong to the set of
        allowed symbols; False otherwise.
    """
    placement = fen.partition(' ')[0]
    allowed = set(allowed_pieces.lower()) | {'k'}
    contains = allowed.__contains__
    for ch in placement:
        if ch.isalpha() and not contains(ch.lower()):
            return False
    return True


def mate_to_cp(
    mate: Optional[int | float],
    max_cp_no_mate: int = 20_000,
    mate_margin: int = 7_000,
    step: int = 100
) -> Optional[int]:
    """Convert mate-in-N notation to centipawn equivalent for unified scoring.
    
    Transforms chess engine mate-in-N values to centipawn (cp) scale for consistent
    model training. Mate values are converted to approximate cp equivalents using
    a linear scaling formula that increases with mate urgency. Garantuje że
    nawet mat w wielu ruchach będzie wyższy niż max_cp_no_mate.
    
    Args:
        mate: Mate-in-N value (positive for advantage, negative for disadvantage).
              None or 0 returns None.
        max_cp_no_mate: Base centipawn value for non-mate positions (default: 20000).
        mate_margin: Additional margin added to mate-in-N conversion (default: 7000).
        step: Centipawn decrement per move to mate (default: 100).
    
    Returns:
        Converted centipawn value (int) or None if mate is None/NaN/0.
    
    Examples:
        mate_to_cp(1) -> 26900 (mate in 1 move)
        mate_to_cp(5) -> 26500 (mate in 5 moves)
        mate_to_cp(100) -> 20100 (mat w 100 ruchach, ale nigdy < max_cp_no_mate)
        mate_to_cp(-3) -> -26700 (opponent mate in 3)
        mate_to_cp(None) -> None
    """
    if mate is None or pd.isna(mate) or mate == 0:
        return None

    sign = 1 if mate > 0 else -1
    cp_value = max_cp_no_mate + mate_margin - step * (abs(int(mate)) - 1)
    return sign * max(abs(cp_value), max_cp_no_mate + step)


def fen_to_tensor(fen: str, always_white_perspective=True) -> np.ndarray:
    """
    Convert FEN to 8x8x12 tensor representation with all piece types.
    
    Args:
        fen: FEN notation string.
        always_white_perspective: If True, board is mirrored for black's turn.
    
    Returns:
        Tensor of shape (8, 8, 12) where channels represent:
        0-5: Friendly pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        6-11: Enemy pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    """
    board = chess.Board(fen)

    if always_white_perspective and not board.turn:
        board = board.mirror()

    tensor = np.zeros((8, 8, 12), dtype=np.uint8)

    for square, piece in board.piece_map().items():
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        row, col = divmod(square, 8)
        tensor[7 - row, col, channel] = 1

    return tensor
