import pytest
from src.data_processing import allowed_pieces_only

def test_empty_fen():
    assert allowed_pieces_only('', '') is True

def test_case_sensitivity():
    assert allowed_pieces_only('8/8/8/8/8/8/8/R7 w - - 0 1', 'r') is True

def test_empty_board_none_allowed_but_king_1():
    assert allowed_pieces_only('8/8/8/8/8/8/8/8 w - - 0 1', '') is True

def test_empty_board_none_allowed_but_king_2():
    assert allowed_pieces_only('8/8/8/8/8/8/8/8 w - - 0 1', 'k') is True

def test_more_allowed_than_on_board():
    assert allowed_pieces_only('1k6/2p5/8/8/8/8/3P4/2K5 w - - 0 1', 'pbr') is True

def test_more_on_board_than_allowed():
    assert allowed_pieces_only('1k6/2p5/8/8/8/8/3P4/2K5 w - - 0 1', '') is False

def test_full_board_none_allowed():
    assert allowed_pieces_only('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', '') is False

def test_full_board_all_allowed():
    assert allowed_pieces_only('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'qrbnp') is True

def test_empty_board_all_allowed():
    assert allowed_pieces_only('8/8/8/8/8/8/8/8 w - - 0 1', 'qrbnp') is True