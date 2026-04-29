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
                # --chinese-rules makes final_score use area scoring, which
                # matches GoGame.score(). Without it GNU Go defaults to
                # Japanese counting and the two scores will disagree by a
                # small constant on most positions.
                [self.gnugo_path, "--mode", "gtp", "--chinese-rules",
                 "--level", str(self.level)],
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
        self._sync_to_history(game)
        my_color = "black" if game.to_move == BLACK else "white"
        res = self._send_command(f"genmove {my_color}")
        if res.startswith("="):
            return self._from_gtp_coords(res[1:].strip())
        return "pass"

    def reset(self) -> None:
        if self._proc:
            self._send_command("clear_board")

    def _sync_to_history(self, game: GoGame) -> None:
        """Replay GoGame.history into GNU Go from a clear board.

        "concede" is not a real GTP move, so we skip it: the engine has
        already marked the game as finished, and GNU Go just needs to see
        the actual stones placed and passes that occurred.
        """
        if self._proc is None:
            return
        self._send_command("clear_board")
        side = BLACK
        for move in game.history:
            if move == "concede":
                continue
            color_str = "black" if side == BLACK else "white"
            self._send_command(f"play {color_str} {self._to_gtp_coords(move)}")
            side = WHITE if side == BLACK else BLACK

    def final_score(self, game: GoGame) -> Optional[dict]:
        """Ask GNU Go for its area-scoring verdict on the current position.

        Returns {"winner": BLACK | WHITE | None, "margin": float, "raw": str}
        or None if the GTP query failed. None winner means jigo (tie).
        """
        if self._proc is None:
            return None
        self._sync_to_history(game)
        res = self._send_command("final_score")
        if not res.startswith("="):
            return None
        text = res[1:].strip()
        if text == "0":
            return {"winner": None, "margin": 0.0, "raw": text}
        if len(text) >= 3 and text[1] == "+":
            head = text[0].upper()
            try:
                margin = float(text[2:])
            except ValueError:
                return None
            winner = BLACK if head == "B" else WHITE if head == "W" else None
            return {"winner": winner, "margin": margin, "raw": text}
        return None

    def __del__(self) -> None:
        if self._proc:
            try:
                self._send_command("quit")
                self._proc.terminate()
            except:
                pass
