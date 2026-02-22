"""
CLI: Download a player's Lichess games and train a style model.

Examples:
    # Basic — downloads up to 2000 rated games across all time controls
    python train_style.py --username magnuscarlsen

    # Blitz games only, more games, custom output path
    python train_style.py --username magnuscarlsen --perf-type blitz --max-games 5000 --output models/magnus_blitz.pt

    # With API token (raises rate limits)
    python train_style.py --username magnuscarlsen --token lip_xxxxxxxxxxxx
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from data.lichess import stream_games
from data.pgn_parser import games_to_dataset
from style.dataset import PlayerGamesDataset
from style.train import train


def main():
    parser = argparse.ArgumentParser(description="Train a style model on a Lichess player's games.")
    parser.add_argument("--username",   required=True,  help="Lichess username")
    parser.add_argument("--token",      default=None,   help="Lichess API token (optional but recommended)")
    parser.add_argument("--max-games",  type=int, default=2000, help="Max games to download (default: 2000)")
    parser.add_argument("--perf-type",  default=None,
                        choices=["bullet", "blitz", "rapid", "classical"],
                        help="Only use games of this time control (default: all)")
    parser.add_argument("--output",     default=None,   help="Where to save the model (.pt). Default: models/<username>_style.pt")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--channels",   type=int, default=128, help="StyleNet channel width")
    parser.add_argument("--res-blocks", type=int, default=6,   help="StyleNet residual block count")
    args = parser.parse_args()

    output = args.output or f"models/{args.username}_style.pt"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ───────────────────────────────────────────────────────
    print(f"\nDownloading games for '{args.username}' from Lichess...")
    games = list(tqdm(
        stream_games(
            username=args.username,
            token=args.token,
            max_games=args.max_games,
            perf_type=args.perf_type,
        ),
        desc="Downloading",
        unit="games",
    ))
    print(f"Downloaded {len(games)} games.")

    if not games:
        print("ERROR: No games found. Check the username and try again.")
        return

    # ── 2. Parse positions ────────────────────────────────────────────────
    print("\nParsing positions...")
    positions_and_moves = games_to_dataset(games, args.username)
    print(f"Extracted {len(positions_and_moves):,} positions from {len(games):,} games.")

    if len(positions_and_moves) < 500:
        print(
            f"WARNING: Only {len(positions_and_moves)} positions found. "
            "Consider downloading more games for better style learning."
        )
    if len(positions_and_moves) < 50:
        print("ERROR: Too few positions to train. Aborting.")
        return

    boards, moves = zip(*positions_and_moves)
    dataset = PlayerGamesDataset(list(boards), list(moves))

    # ── 3. Train ──────────────────────────────────────────────────────────
    print(f"\nTraining StyleNet for '{args.username}'...")
    train(
        dataset=dataset,
        model_path=output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        channels=args.channels,
        num_res_blocks=args.res_blocks,
    )

    print(f"\nAll done! To use this model:")
    print(f"  python play.py --model {output} --elo 1900")


if __name__ == "__main__":
    main()
