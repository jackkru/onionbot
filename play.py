"""
CLI: Ask OnionBot to play a move from a given position.

Examples:
    # Starting position
    python play.py --model models/myusername_style.pt

    # Specific FEN
    python play.py --model models/myusername_style.pt --fen "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

    # Pure Stockfish at 2200, no style
    python play.py --model models/myusername_style.pt --style-weight 0 --elo 2200

    # Heavy style influence
    python play.py --model models/myusername_style.pt --style-weight 0.8 --elo 1900
"""

import argparse
import chess

from config import BotConfig
from bot import OnionBot


def main():
    parser = argparse.ArgumentParser(description="Play a move with OnionBot.")
    parser.add_argument("--model",        required=True,       help="Path to style model (.pt)")
    parser.add_argument("--fen",          default=None,        help="FEN string (default: starting position)")
    parser.add_argument("--elo",          type=int, default=2000, help="Stockfish Elo limit (default: 2000)")
    parser.add_argument("--style-weight", type=float, default=0.5, help="Style influence 0–1 (default: 0.5)")
    parser.add_argument("--top-n",        type=int, default=5,  help="Engine candidates to consider (default: 5)")
    parser.add_argument("--temperature",  type=float, default=0.5, help="Sampling temperature (0 = always top move)")
    parser.add_argument("--stockfish",    default=None,        help="Path to Stockfish binary (auto-detected if omitted)")
    args = parser.parse_args()

    config = BotConfig(
        stockfish_path=args.stockfish,
        engine_elo=args.elo,
        model_path=args.model,
        style_weight=args.style_weight,
        top_n=args.top_n,
        temperature=args.temperature,
    )

    board = chess.Board(args.fen) if args.fen else chess.Board()

    print("\nPosition:")
    print(board)
    print(f"\nFEN: {board.fen()}")
    print(f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}")

    bot = OnionBot(config)
    move = bot.get_move(board)

    print(f"\nOnionBot plays: {board.san(move)}  ({move.uci()})")


if __name__ == "__main__":
    main()
