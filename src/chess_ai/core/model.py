"""
The main wrapper class for the Keras model.
Responsible for loading the model, predicting moves, and evaluating positions.
"""
from typing import List

import chess
from chess import Board, Move
from tensorflow.keras.models import load_model


class ChessAI:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load_model(model_path)

    def predict_move(self, board: Board) -> Move:
        legal_moves = board.generate_legal_moves()

        # 2. Prosi o listę legalnych ruchów. Będzie ich "n".
        # 3. Konwertuje n-ruchów do n-pozycji FEN.
        # 4. Konwertuje wszystkie FEN na tensory.
        # 5. Ocenia wszystkie legalne ruchy
        # 6. Wybiera najlepszy ruch.
        # 7. Zwraca najlepszy ruch.
        pass

    def get_list_of_boards_with_legal_moves(self, board: Board, legal_moves: List[Move]) -> List[Board]:
        list_of_boards = []
        for move in legal_moves:
            board.push(move)
            list_of_boards.append(board)
            board.pop()
        return list_of_boards

    def evaluate_position(self, fen: str) -> float:
        pass
