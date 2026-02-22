"""
Lichess API interface.
Downloads games for a given username using the berserk client library.
"""

import berserk
from typing import Generator, Dict, Optional


def get_client(token: Optional[str] = None) -> berserk.Client:
    if token:
        session = berserk.TokenSession(token)
        return berserk.Client(session=session)
    return berserk.Client()


def stream_games(
    username: str,
    token: Optional[str] = None,
    max_games: int = 5000,
    perf_type: Optional[str] = None,
    rated_only: bool = True,
) -> Generator[Dict, None, None]:
    """
    Stream games for a Lichess user as dicts.

    Each dict contains:
      - 'moves': space-separated UCI move string  (e.g. "e2e4 e7e5 g1f3 ...")
      - 'players': {'white': {'user': {'name': ...}}, 'black': ...}
      - 'winner': 'white' | 'black' | None (draw)
      - 'status': game termination reason

    Args:
        username:   Lichess username to download games for.
        token:      Optional API token (raises rate limits, required for private games).
        max_games:  Maximum number of games to download.
        perf_type:  Filter by time control: 'bullet' | 'blitz' | 'rapid' | 'classical'.
        rated_only: Only include rated games (default True, unrated games are noisier).
    """
    client = get_client(token)

    kwargs: Dict = {
        "max": max_games,
        "opening": False,
        "clocks": False,
        "evals": False,
        "rated": rated_only,
    }
    if perf_type:
        kwargs["perf_type"] = perf_type

    yield from client.games.export_by_player(username, **kwargs)
