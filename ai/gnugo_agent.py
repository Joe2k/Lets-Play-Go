"""
GnuGoAgent: An agent that plays using the GNU Go engine via GTP.
"""

from __future__ import annotations

import subprocess
import os
import sys
from typing import Optional, Union

from engine.go_engine import GoGame, BLACK, WHITE, SIZE

Move = Union[tuple[int, int], str]  # (r, c) or "pass"

class GnuGoAgent:
    def __init__(
        self,
        level: int = 8,
        gnugo_path: str = "gnugo",
    ) -> None:
        self.level = level
        self.gnugo_path = gnugo_path
        self._proc: Optional[subprocess.Popen] = None
        self._start_gnugo()

    def _start_gnugo(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [self.gnugo_path, "--mode", "gtp", "--level", str(self.level)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Initialize board
            self._send_command(f"boardsize {SIZE}")
            self._send_command(f"komi 2.5")
        except FileNotFoundError:
            print(f"Error: {self.gnugo_path} not found. Please install GNU Go.", file=sys.stderr)
            self._proc = None

    def _send_command(self, cmd: str) -> str:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return ""
        
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()
        
        response = ""
        while True:
            line = self._proc.stdout.readline().strip()
            if not line and response:
                break
            if line:
                response += line + "\n"
            if line.startswith("=") or line.startswith("?"):
                # End of response (GTP protocol uses double newline)
                # But sometimes it's faster to just read until we get the status
                pass
            
        # Proper GTP response ends with an empty line
        # The first line starts with = or ?
        return response.strip()

    def _to_gtp_coords(self, move: Move) -> str:
        if move == "pass":
            return "pass"
        row, col = move
        col_char = "ABCDEFGHJ"[col]
        row_num = SIZE - row
        return f"{col_char}{row_num}"

    def _from_gtp_coords(self, gtp_move: str) -> Move:
        gtp_move = gtp_move.upper().strip()
        if gtp_move == "PASS":
            return "pass"
        
        col_char = gtp_move[0]
        row_num = int(gtp_move[1:])
        
        col = "ABCDEFGHJ".index(col_char)
        row = SIZE - row_num
        return (row, col)

    def select_move(self, game: GoGame) -> Move:
        if self._proc is None:
            return "pass"

        # Sync board state using history
        self._send_command("clear_board")
        
        current_to_move = BLACK
        for move in game.history:
            c_str = "black" if current_to_move == BLACK else "white"
            self._send_command(f"play {c_str} {self._to_gtp_coords(move)}")
            current_to_move = WHITE if current_to_move == BLACK else BLACK

        my_color = "black" if game.to_move == BLACK else "white"
        res = self._send_command(f"genmove {my_color}")
        
        if res.startswith("="):
            move_str = res[1:].strip()
            return self._from_gtp_coords(move_str)
        
        return "pass"

    def reset(self) -> None:
        if self._proc:
            self._send_command("clear_board")

    def __del__(self) -> None:
        if self._proc:
            try:
                self._send_command("quit")
                self._proc.terminate()
            except:
                pass
