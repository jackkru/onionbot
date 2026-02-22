"""
PyTorch Dataset that holds (board_tensor, move_index) pairs for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import chess

from style.features import board_to_tensor, move_to_index


class PlayerGamesDataset(Dataset):
    """
    Dataset of board positions and the move a specific player made in each one.

    Args:
        boards: List of chess.Board objects (snapshots before the move).
        moves:  Corresponding list of chess.Move objects that were played.
    """

    def __init__(self, boards: list, moves: list):
        assert len(boards) == len(moves), "boards and moves must have the same length"

        n = len(boards)
        self._tensors = np.zeros((n, *board_to_tensor(chess.Board()).shape), dtype=np.float32)
        self._labels  = np.zeros(n, dtype=np.int64)

        for i, (board, move) in enumerate(zip(boards, moves)):
            self._tensors[i] = board_to_tensor(board)
            self._labels[i]  = move_to_index(move)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._tensors[idx]),
            int(self._labels[idx]),
        )
