from dataclasses import dataclass
from typing import Optional


@dataclass
class BotConfig:
    # Stockfish settings
    stockfish_path: Optional[str] = None  # auto-detect if None
    engine_elo: Optional[int] = 2000      # None = full strength

    # Style settings
    model_path: Optional[str] = None
    style_weight: float = 0.5             # 0 = pure engine, 1 = pure style
    top_n: int = 5                        # top N engine moves to re-rank
    temperature: float = 0.5             # sampling temperature (0 = argmax, always play top move)

    # Model architecture (must match what was used at training time)
    model_channels: int = 128
    model_res_blocks: int = 6

    # Timing
    time_per_move: float = 0.1           # seconds Stockfish gets per move

    # Lichess
    lichess_token: Optional[str] = None
