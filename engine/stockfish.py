"""
Thin wrapper around Stockfish via python-chess's UCI engine interface.

Usage:
    with StockfishEngine(elo=1900) as engine:
        moves = engine.get_top_moves(board, n=5)
        best  = moves[0].move
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

import chess
import chess.engine


@dataclass
class ScoredMove:
    move:     chess.Move
    score_cp: int        # centipawns from the perspective of the side to move
    pv:       List[chess.Move] = field(default_factory=list)


class StockfishEngine:
    """
    Context manager that opens a Stockfish process and exposes get_top_moves().

    Args:
        path:    Path to stockfish binary. Auto-detected if None.
        elo:     Target playing strength (1320–3190). None = full strength.
        threads: Number of CPU threads for Stockfish.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        elo: Optional[int] = None,
        threads: int = 1,
    ):
        self.path    = path or self._find_stockfish()
        self.elo     = elo
        self.threads = threads
        self._engine: Optional[chess.engine.SimpleEngine] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def __enter__(self) -> "StockfishEngine":
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        options: dict = {"Threads": self.threads}
        if self.elo is not None:
            options["UCI_LimitStrength"] = True
            options["UCI_Elo"] = max(1320, min(3190, self.elo))
        self._engine.configure(options)
        return self

    def __exit__(self, *_) -> None:
        if self._engine:
            self._engine.quit()
            self._engine = None

    # ── core method ───────────────────────────────────────────────────────

    def get_top_moves(
        self,
        board: chess.Board,
        n: int = 5,
        time_limit: float = 0.1,
    ) -> List[ScoredMove]:
        """
        Ask Stockfish for its top N moves in the given position.

        Returns a list of ScoredMove sorted best-first (highest score first
        from the perspective of the side to move).

        Args:
            board:      Position to analyse.
            n:          Number of moves to return (MultiPV).
            time_limit: Seconds Stockfish gets to think.
        """
        if self._engine is None:
            raise RuntimeError("Must be used as a context manager.")
        if board.is_game_over():
            return []

        n = min(n, board.legal_moves.count())
        infos = self._engine.analyse(
            board,
            chess.engine.Limit(time=time_limit),
            multipv=n,
        )

        results: List[ScoredMove] = []
        for info in infos:
            pv = info.get("pv", [])
            if not pv:
                continue
            score_obj = info["score"].pov(board.turn)
            if score_obj.is_mate():
                cp = 10_000 if score_obj.mate() > 0 else -10_000
            else:
                cp = score_obj.score(mate_score=10_000)
            results.append(ScoredMove(move=pv[0], score_cp=cp, pv=pv))

        # Sort best-first (highest cp first)
        results.sort(key=lambda x: x.score_cp, reverse=True)
        return results

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _find_stockfish() -> str:
        sf = shutil.which("stockfish")
        if sf:
            return sf
        for candidate in [
            "/opt/homebrew/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
        ]:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "Stockfish not found. Install it with:\n"
            "  macOS: brew install stockfish\n"
            "  Linux: sudo apt install stockfish"
        )
