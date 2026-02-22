"""
OnionBot: the top-level chess bot class.

Ties together Stockfish + StyleNet and exposes a single get_move() method.
"""

import os
from typing import Optional

import chess
import torch

from config import BotConfig
from engine.stockfish import StockfishEngine
from engine.blend import blend_and_select
from style.model import StyleNet


class OnionBot:
    """
    Chess bot that blends Stockfish evaluation with a learned player style.

    Usage:
        config = BotConfig(model_path="models/myusername_style.pt", engine_elo=1900)
        bot = OnionBot(config)
        move = bot.get_move(board)
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        self.style_model: Optional[StyleNet] = None
        self._load_style_model()

    def _load_style_model(self) -> None:
        path = self.config.model_path
        if not path or not os.path.exists(path):
            print("No style model loaded — will use pure Stockfish.")
            return

        self.style_model = StyleNet(
            channels=self.config.model_channels,
            num_res_blocks=self.config.model_res_blocks,
        ).to(self.device)
        self.style_model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.style_model.eval()
        print(f"Style model loaded from {path}")

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Return the bot's chosen move for the given position.

        Opens a fresh Stockfish process each call (cheap, avoids stale state).
        """
        if board.is_game_over():
            raise ValueError("Game is already over.")

        with StockfishEngine(
            path=self.config.stockfish_path,
            elo=self.config.engine_elo,
        ) as engine:
            # No style model → pure engine
            if self.style_model is None or self.config.style_weight == 0.0:
                candidates = engine.get_top_moves(board, n=1, time_limit=self.config.time_per_move)
                return candidates[0].move if candidates else next(iter(board.legal_moves))

            return blend_and_select(
                board=board,
                engine=engine,
                style_model=self.style_model,
                device=self.device,
                top_n=self.config.top_n,
                style_weight=self.config.style_weight,
                time_limit=self.config.time_per_move,
                temperature=self.config.temperature,
            )
