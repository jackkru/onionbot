#!/usr/bin/env bash
# Launcher for chess GUIs (Arena, Cutechess, etc.)
# Point your GUI to this script as the engine executable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/uci.py" "$@"
