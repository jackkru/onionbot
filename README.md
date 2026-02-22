# OnionBot

A chess engine that learns to play like a specific person.

Point it at any Lichess username, it downloads their games, trains a neural network on their move choices, and produces a bot that plays in their style — same opening tendencies, same positional preferences, same tactical habits. The underlying engine is Stockfish (so it won't blunder pieces), but the move selection is biased toward what *that player* would do.

---

## How it works

There are two layers:

**1. Stockfish** handles tactical correctness. Given a position, it finds the top N candidate moves and scores them objectively.

**2. StyleNet** handles personality. A small residual CNN is trained via imitation learning on the target player's games — thousands of `(board position → move played)` pairs. At inference time it outputs a probability for each candidate move: *how likely is this player to make this move in a position like this?*

The final move is chosen by blending the two scores:

```
final_score = (1 - style_weight) * engine_score + style_weight * style_score
```

`style_weight = 0` → pure Stockfish. `style_weight = 1` → pure imitation. The sweet spot for "plays like them but doesn't blunder" is around 0.4–0.6.

---

## Setup

**Requirements:** Python 3.11+, Stockfish

```bash
# Install Stockfish
brew install stockfish        # macOS
sudo apt install stockfish    # Ubuntu/Debian

# Clone and set up Python environment
git clone https://github.com/jackkru/onionbot.git
cd onionbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training a style model

```bash
source .venv/bin/activate

# Basic: download up to 2000 games across all time controls
python train_style.py --username lichess_username

# Recommended: more games, blitz only (higher quality moves than bullet)
python train_style.py --username lichess_username --max-games 5000 --perf-type blitz

# With a Lichess API token (higher rate limits, required for private games)
python train_style.py --username lichess_username --token lip_xxxxxxxxxxxx
```

The model is saved to `models/<username>_style.pt`. Training on 5000 games takes about 15 minutes on Apple Silicon.

**What the accuracy numbers mean:**

The network is trained to predict the exact move the player made. ~30–35% top-1 accuracy on held-out games is strong — random guessing from ~30 legal moves would give ~3%. Higher accuracy isn't necessarily better; it can mean overfitting to specific positions rather than learning transferable tendencies.

---

## Playing a move

```bash
source .venv/bin/activate

# Ask the bot for a move from the starting position
python play.py --model models/lichess_username_style.pt

# From a specific position (FEN)
python play.py --model models/lichess_username_style.pt \
    --fen "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

# Tune the style vs. engine balance
python play.py --model models/lichess_username_style.pt --style-weight 0.7 --elo 1900

# Pure Stockfish (no style)
python play.py --model models/lichess_username_style.pt --style-weight 0
```

---

## Using in a chess GUI

OnionBot implements the UCI protocol. Point any UCI-compatible chess GUI (Arena, Cutechess, Banksia, etc.) at `onionbot.sh`:

```
Engine executable: /path/to/onionbot/onionbot.sh
```

### UCI options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `StyleModel` | string | _(none)_ | Path to trained `.pt` file |
| `StyleWeight` | 0–100 | 50 | Style influence. 0 = pure Stockfish, 100 = pure imitation |
| `EngineElo` | 1320–3190 | 2000 | Stockfish strength cap |
| `FullStrength` | bool | false | Ignore Elo cap, use full Stockfish strength |
| `TopN` | 1–10 | 5 | Candidate moves to re-rank |
| `Temperature` | 0–100 | 50 | 0 = always play top move, higher = more variety |

**On strength:** `EngineElo` and `StyleWeight` are independent axes. A low Elo with high style weight produces the most faithful imitation — objectively weaker but more characteristic. Full strength with moderate style produces a stronger player with stylistic tendencies.

---

## Project structure

```
onionbot/
├── data/
│   ├── lichess.py       # Lichess API — stream games for a username
│   └── pgn_parser.py    # Replay games → (board, move) pairs
├── style/
│   ├── features.py      # Board → 18-plane tensor + move encoding
│   ├── model.py         # StyleNet: ResNet policy network
│   ├── dataset.py       # PyTorch Dataset
│   └── train.py         # Training loop
├── engine/
│   ├── stockfish.py     # Stockfish subprocess wrapper
│   └── blend.py         # Combine engine + style scores
├── bot.py               # OnionBot: top-level class
├── uci.py               # UCI protocol interface
├── onionbot.sh          # Shell launcher for chess GUIs
├── train_style.py       # CLI: download games + train
├── play.py              # CLI: get a move from a position
└── config.py            # BotConfig dataclass
```

---

## Limitations & future work

- **Bullet games are noisy.** Moves made on increment aren't always intentional. Training on blitz or rapid games gives cleaner style signal.
- **Openings aren't modeled separately.** The network learns opening tendencies implicitly from the data. An explicit opening book built from the player's games would make the opening phase more faithful.
- **Style generalises imperfectly.** The network has never seen every possible position. It extrapolates from structural similarities, which works well for common patterns but can be unreliable in unusual positions.
- **Underpromotions are ignored.** All promotion moves are treated as queen promotions. This affects maybe 1 in 10,000 positions.
