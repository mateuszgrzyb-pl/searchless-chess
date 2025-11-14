"""Tests for `data_processing` module"""
import chess
import numpy as np

from src.data_processing import (
    allowed_pieces_only,
    mate_to_cp,
    fen_to_tensor
)

# %% `allowed_pieces_only`
def test_empty_fen():
    """Empty FEN and allowed pieces should return True."""
    assert allowed_pieces_only('', '') is True

def test_case_sensitivity():
    """Lowercase 'r' should match uppercase 'R' on board (case-insensitive)."""
    assert allowed_pieces_only('8/8/8/8/8/8/8/R7 w - - 0 1', 'r') is True

def test_empty_board_none_allowed_but_king_1():
    """Empty board with no allowed pieces should return True (king always allowed)."""
    assert allowed_pieces_only('8/8/8/8/8/8/8/8 w - - 0 1', '') is True

def test_empty_board_none_allowed_but_king_2():
    """Empty board with 'k' allowed should return True."""
    assert allowed_pieces_only('8/8/8/8/8/8/8/8 w - - 0 1', 'k') is True

def test_more_allowed_than_on_board():
    """More allowed pieces than on board should return True."""
    assert allowed_pieces_only('1k6/2p5/8/8/8/8/3P4/2K5 w - - 0 1', 'pbr') is True

def test_more_on_board_than_allowed():
    """Board with pieces not in allowed list should return False."""
    assert allowed_pieces_only('1k6/2p5/8/8/8/8/3P4/2K5 w - - 0 1', '') is False

def test_full_board_none_allowed():
    """Full starting position with no allowed pieces should return False."""
    assert allowed_pieces_only('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', '') is False

def test_full_board_all_allowed():
    """Full starting position with all pieces allowed should return True."""
    assert allowed_pieces_only('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'qrbnp') is True

def test_empty_board_all_allowed():
    """Empty board with all pieces allowed should return True."""
    assert allowed_pieces_only('8/8/8/8/8/8/8/8 w - - 0 1', 'qrbnp') is True

# %% `mate_to_cp`
def test_mate_in_1():
    """Mate in 1 should return 27_000"""
    assert mate_to_cp(1) == 27000

def test_mate_in_5():
    """Mate in 5 should return 26_600"""
    assert mate_to_cp(5) == 26600

def test_mate_in_100_step_100():
    """
    Mate in 100 should not return less than `max_cp_no_mate`.
    Minimum value for situation with mate is `max_cp_no_mate` + `step`
    """
    assert mate_to_cp(mate=100, max_cp_no_mate=20000, step=100) == 20100

def test_mate_in_100_step_200():
    """
    Mate in 100 should not return less than `max_cp_no_mate`.
    Minimum value for situation with mate is `max_cp_no_mate` + `step`
    """
    assert mate_to_cp(mate=100, max_cp_no_mate=20000, step=200) == 20200

# %% `fen_to_tensor`
def test_starting_position_shape_and_type():
    """Verify that the tensor for the starting position has correct shape and data type."""
    tensor = fen_to_tensor(chess.STARTING_FEN)
    assert tensor.shape == (8, 8, 12)
    assert tensor.dtype == np.uint8

def test_white_pawn_placement_on_e2():
    """Verify that the white pawn on e2 is correctly placed in the tensor."""
    tensor = fen_to_tensor(chess.STARTING_FEN)
    # Square e2 is (row 1, column 4) in chess notation.
    # In numpy this is index [6, 4]. Channel for white pawn is 0.
    assert tensor[6, 4, 0] == 1

def test_black_knight_placement_on_g8():
    """Verify that the black knight on g8 is correctly placed in the tensor."""
    tensor = fen_to_tensor(chess.STARTING_FEN)
    # Square g8 is (row 7, column 6) in chess notation.
    # In numpy this is index [0, 6]. Channel for black knight is 7.
    assert tensor[0, 6, 7] == 1

def test_empty_square_e4_is_empty():
    """Verify that empty square e4 has no pieces (all channels are 0)."""
    tensor = fen_to_tensor(chess.STARTING_FEN)
    # Square e4 in numpy is index [4, 4].
    # Sum of values across all 12 channels for this square should be 0.
    assert np.sum(tensor[4, 4, :]) == 0

def test_total_piece_count_is_correct():
    """Verify that the sum of all values in the tensor matches the number of pieces."""
    # Starting position has 32 pieces.
    tensor = fen_to_tensor(chess.STARTING_FEN)
    assert np.sum(tensor) == 32

    # Endgame with two pieces.
    endgame_fen = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
    tensor_endgame = fen_to_tensor(endgame_fen)
    assert np.sum(tensor_endgame) == 2

def test_black_turn_with_white_perspective():
    """
    Verify that the board is correctly mirrored when it's black's turn.
    """
    # FEN after 1. e4 e5 (it's black's turn to move)
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    tensor = fen_to_tensor(fen, always_white_perspective=True)

    # After `board.mirror()`, the black pawn from e5 is moved to e4 and its color is flipped to white.
    # We assert that square e4 (index [4, 4]) now contains a white pawn (channel 0).
    assert tensor[4, 4, 0] == 1

    # Simultaneously, the original white pawn from e4 is moved to e5 and its color is flipped to black.
    # We assert that square e5 (index [3, 4]) now contains a black pawn (channel 6).
    assert tensor[3, 4, 6] == 1

def test_black_turn_without_white_perspective():
    """
    Verify that the board is NOT mirrored when `always_white_perspective=False`.
    """
    # Same FEN as in the test above, but with different flag.
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    tensor = fen_to_tensor(fen, always_white_perspective=False)

    # Board is not mirrored. Black pawn is on e5 (index [3, 4]).
    # Channel for black pawn is 6.
    assert tensor[3, 4, 6] == 1

    # Square e4 (index [4, 4]) should be occupied by white pawn (channel 0).
    assert tensor[4, 4, 0] == 1
