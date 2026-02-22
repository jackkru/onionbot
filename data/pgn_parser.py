"""
Converts raw Lichess game dicts into (chess.Board, chess.Move) training pairs.

Only moves made by the target player are kept — we're learning their decisions,
not their opponent's.
"""

import chess
from typing import List, Tuple, Dict, Optional


def game_to_positions(
    game: Dict,
    username: str,
) -> List[Tuple[chess.Board, chess.Move]]:
    """
    Replay a single game and collect (board_state, move_played) pairs
    for every move the target player made.

    Args:
        game:     Game dict from Lichess API (must have 'moves' and 'players' fields).
        username: The target player's Lichess username.

    Returns:
        List of (board, move) tuples. Board is a snapshot BEFORE the move was played.
    """
    if not game.get("moves"):
        return []

    players = game.get("players", {})
    white_name = players.get("white", {}).get("user", {}).get("name", "").lower()
    black_name = players.get("black", {}).get("user", {}).get("name", "").lower()

    if username.lower() == white_name:
        target_color = chess.WHITE
    elif username.lower() == black_name:
        target_color = chess.BLACK
    else:
        return []  # Target player not in this game

    board = chess.Board()
    positions: List[Tuple[chess.Board, chess.Move]] = []

    for move_str in game["moves"].split():
        try:
            # Lichess NDJSON returns SAN notation (e.g. "Nf6", "Bxc3+")
            move = board.parse_san(move_str)
            if board.turn == target_color:
                positions.append((board.copy(), move))
            board.push(move)
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            break  # Corrupted or unsupported position — stop replaying this game

    return positions


def games_to_dataset(
    games: List[Dict],
    username: str,
) -> List[Tuple[chess.Board, chess.Move]]:
    """
    Process a list of game dicts into a flat list of (board, move) pairs.

    Args:
        games:    List of game dicts from Lichess API.
        username: The target player's Lichess username.

    Returns:
        All (board, move) pairs from the player's perspective across all games.
    """
    all_positions: List[Tuple[chess.Board, chess.Move]] = []
    for game in games:
        all_positions.extend(game_to_positions(game, username))
    return all_positions
