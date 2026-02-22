"""
Blends Stockfish engine scores with StyleNet probabilities to pick a move.

The blending formula is:

    final_score = (1 - style_weight) * engine_prob
                +      style_weight  * style_prob

where both engine_prob and style_prob are softmax-normalised over the top-N
candidates so they live on the same scale.

style_weight = 0.0  →  pure Stockfish (plays objectively best move)
style_weight = 1.0  →  pure style     (plays most characteristic move regardless of quality)
style_weight = 0.5  →  balanced       (recommended starting point)

temperature controls how deterministically the final move is selected:
    temperature = 0  →  always return the top-scoring move (argmax)
    temperature > 0  →  sample proportionally (adds variety, more human-like)
"""

from typing import List, Optional

import numpy as np
import torch

import chess

from engine.stockfish import StockfishEngine, ScoredMove
from style.model import StyleNet
from style.features import board_to_tensor, move_to_index, legal_move_indices


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def blend_and_select(
    board: chess.Board,
    engine: StockfishEngine,
    style_model: StyleNet,
    device: torch.device,
    top_n: int = 5,
    style_weight: float = 0.5,
    time_limit: float = 0.1,
    temperature: float = 0.5,
) -> chess.Move:
    """
    Select a move by combining engine evaluation and style network preferences.

    Args:
        board:        Current board position.
        engine:       Open StockfishEngine context manager.
        style_model:  Trained StyleNet (in eval mode).
        device:       Torch device the model lives on.
        top_n:        How many engine candidates to consider.
        style_weight: Mix ratio (0 = pure engine, 1 = pure style).
        time_limit:   Seconds Stockfish gets to think.
        temperature:  Sampling temperature (0 = argmax).

    Returns:
        The chosen chess.Move.
    """
    # ── 1. Get engine candidates ──────────────────────────────────────────
    candidates: List[ScoredMove] = engine.get_top_moves(board, n=top_n, time_limit=time_limit)

    if not candidates:
        return next(iter(board.legal_moves))

    if len(candidates) == 1 or style_weight == 0.0:
        return candidates[0].move

    moves = [c.move for c in candidates]

    # Normalise engine scores to a probability distribution
    engine_scores = np.array([c.score_cp for c in candidates], dtype=np.float32)
    engine_probs  = _softmax(engine_scores / 100.0)   # divide by 100 → pawn units

    # ── 2. Get style scores for each candidate ────────────────────────────
    board_tensor = (
        torch.from_numpy(board_to_tensor(board))
        .unsqueeze(0)                                  # (1, 18, 8, 8)
        .to(device)
    )
    legal_indices = legal_move_indices(board)

    with torch.no_grad():
        probs_tensor = style_model.move_probs(board_tensor, legal_indices)  # (4096,)

    style_scores = np.array(
        [probs_tensor[move_to_index(m)].item() for m in moves],
        dtype=np.float32,
    )
    # Re-normalise over just the candidates (they may not sum to 1 already)
    style_probs = style_scores / (style_scores.sum() + 1e-10)

    # ── 3. Blend ──────────────────────────────────────────────────────────
    blended = (1.0 - style_weight) * engine_probs + style_weight * style_probs

    # ── 4. Select ─────────────────────────────────────────────────────────
    if temperature <= 0.0:
        return moves[int(blended.argmax())]

    # Sharpen/flatten the distribution with temperature then sample
    logits = np.log(blended + 1e-10) / temperature
    probs  = _softmax(logits)
    return moves[np.random.choice(len(moves), p=probs)]
