"""
UCI (Universal Chess Interface) protocol implementation for OnionBot.

This is the entry point when loading OnionBot into any chess GUI
(Arena, Cutechess, Lichess board, etc.).

Usage:
    python uci.py

The GUI will communicate via stdin/stdout using the UCI protocol.
Point your GUI to this script (or the onionbot.sh launcher wrapper).

Exposed UCI options:
    StyleModel   — path to a trained .pt file
    StyleWeight  — 0 (pure engine) to 100 (pure style), default 50
    EngineElo    — Stockfish strength limit 1320–3190, default 2000
    FullStrength — if true, ignores EngineElo and plays at full strength
    TopN         — how many engine candidates to consider, default 5
    Temperature  — 0 = always pick top move, 100 = sample more randomly
"""

import sys
import threading
import chess

from config import BotConfig
from bot import OnionBot


class UCIEngine:
    def __init__(self):
        self.board          = chess.Board()
        self.config         = BotConfig()
        self._bot: OnionBot | None = None
        self._stop          = threading.Event()
        self._search_thread: threading.Thread | None = None

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            self._dispatch(line.strip())
        # stdin closed — wait for any in-flight search before exiting
        if self._search_thread and self._search_thread.is_alive():
            self._search_thread.join(timeout=10)

    # ── command dispatcher ────────────────────────────────────────────────

    def _dispatch(self, line: str) -> None:
        if not line:
            return
        tokens = line.split()
        cmd    = tokens[0]

        if   cmd == "uci":        self._cmd_uci()
        elif cmd == "isready":    self._cmd_isready()
        elif cmd == "setoption":  self._cmd_setoption(tokens)
        elif cmd == "ucinewgame": self._cmd_ucinewgame()
        elif cmd == "position":   self._cmd_position(tokens)
        elif cmd == "go":         self._cmd_go(tokens)
        elif cmd == "stop":       self._stop.set()
        elif cmd == "quit":       self._cmd_quit()
        # unknown commands are silently ignored (spec-compliant)

    # ── uci ───────────────────────────────────────────────────────────────

    def _cmd_uci(self) -> None:
        self._send("id name OnionBot")
        self._send("id author jackkru")
        self._send("")
        self._send("option name StyleModel   type string  default ")
        self._send("option name StyleWeight  type spin    default 50  min 0   max 100")
        self._send("option name EngineElo    type spin    default 2000 min 1320 max 3190")
        self._send("option name FullStrength type check   default false")
        self._send("option name TopN         type spin    default 5   min 1   max 10")
        self._send("option name Temperature  type spin    default 50  min 0   max 100")
        self._send("uciok")

    def _cmd_isready(self) -> None:
        # Lazily initialise the bot on the first isready so the GUI doesn't
        # time out while we load the model weights.
        self._ensure_bot()
        self._send("readyok")

    # ── setoption ─────────────────────────────────────────────────────────

    def _cmd_setoption(self, tokens: list) -> None:
        # setoption name <Name> value <Value>
        try:
            ni = tokens.index("name")  + 1
            vi = tokens.index("value") + 1
        except ValueError:
            return

        name  = " ".join(tokens[ni : vi - 1]).lower()
        value = " ".join(tokens[vi:])

        if name == "stylemodel":
            self.config.model_path = value.strip() or None
            self._bot = None                          # force model reload

        elif name == "styleweight":
            self.config.style_weight = int(value) / 100.0

        elif name == "engineelo":
            v = int(value)
            self.config.engine_elo = v if v > 0 else None

        elif name == "fullstrength":
            if value.lower() == "true":
                self.config.engine_elo = None

        elif name == "topn":
            self.config.top_n = int(value)

        elif name == "temperature":
            self.config.temperature = int(value) / 100.0

    # ── ucinewgame ────────────────────────────────────────────────────────

    def _cmd_ucinewgame(self) -> None:
        self.board = chess.Board()

    # ── position ──────────────────────────────────────────────────────────

    def _cmd_position(self, tokens: list) -> None:
        """
        position startpos [moves e2e4 e7e5 ...]
        position fen <FEN> [moves ...]
        """
        self.board = chess.Board()

        if "fen" in tokens:
            fi = tokens.index("fen") + 1
            fe = tokens.index("moves") if "moves" in tokens else len(tokens)
            self.board = chess.Board(" ".join(tokens[fi:fe]))

        if "moves" in tokens:
            for uci in tokens[tokens.index("moves") + 1 :]:
                try:
                    self.board.push_uci(uci)
                except ValueError:
                    break

    # ── go ────────────────────────────────────────────────────────────────

    def _cmd_go(self, tokens: list) -> None:
        """Parse time controls and start the search thread."""
        self._apply_time_controls(tokens)
        self._stop.clear()
        self._search_thread = threading.Thread(target=self._search, daemon=True)
        self._search_thread.start()

    def _cmd_quit(self) -> None:
        """Wait for any in-flight search to finish, then exit cleanly."""
        if self._search_thread and self._search_thread.is_alive():
            self._stop.set()
            self._search_thread.join(timeout=5)
        sys.exit(0)

    def _apply_time_controls(self, tokens: list) -> None:
        if "movetime" in tokens:
            idx = tokens.index("movetime")
            self.config.time_per_move = int(tokens[idx + 1]) / 1000.0
            return

        key = "wtime" if self.board.turn == chess.WHITE else "btime"
        if key in tokens:
            idx           = tokens.index(key)
            remaining_ms  = int(tokens[idx + 1])
            # Use ~2% of remaining time, clamped to [50ms, 2s]
            self.config.time_per_move = max(0.05, min(2.0, remaining_ms / 1000 * 0.02))

    # ── search thread ─────────────────────────────────────────────────────

    def _search(self) -> None:
        self._ensure_bot()
        board_snapshot = self.board.copy()

        try:
            move = self._bot.get_move(board_snapshot)
        except Exception as e:
            self._send(f"info string error: {e}")
            move = next(iter(board_snapshot.legal_moves), None)

        if move is None:
            return

        self._send(f"bestmove {move.uci()}")

    # ── helpers ───────────────────────────────────────────────────────────

    def _ensure_bot(self) -> None:
        if self._bot is None:
            self._bot = OnionBot(self.config)

    @staticmethod
    def _send(msg: str) -> None:
        print(msg, flush=True)


if __name__ == "__main__":
    UCIEngine().run()
