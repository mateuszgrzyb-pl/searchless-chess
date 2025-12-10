"""
The main wrapper class for the Keras model.
Responsible for loading the model, predicting moves, and evaluating positions.
"""
from typing import Optional

import numpy as np
from chess import Board, Move
from keras.models import load_model
import tensorflow as tf

from src.data_preparation.data_processing import fen_to_tensor


class ChessAI:
    """
    Class to manage AI chess bot.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load_model(model_path, compile=False)

    @tf.function(reduce_retracing=True)
    def predict(self, input_tensor):
        """Fast Keras prediction"""
        return self.model(input_tensor, training=False)

    def make_move(self, board: Board) -> Optional[Move]:
        """
        Predicts which move will be the best based on model evaluation.
        Returns None if no moves are possible (game over).
        """
        # 1. Get list of legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        # 2. Creates list of tensors for every legal move
        input_tensors = []
        for move in legal_moves:
            board.push(move)
            fen = board.fen()
            tensor = fen_to_tensor(fen, always_white_perspective=True)
            input_tensors.append(tensor)
            board.pop() # Cancel move to get previous board state

        tensors_batch = np.array(input_tensors)

        # 3. Prediction for all moves (Batch Prediction).
        evaluations = self.predict(tensors_batch)
        evaluations = evaluations.numpy().flatten()

        # 4. Choosing the best move.
        best_move_id = np.argmin(evaluations)

        return legal_moves[best_move_id]
