"""
9x9 Go rules engine.

Board is stored as a flat list of 81 ints (EMPTY/BLACK/WHITE).

Rule order inside place_stone matters:
  capture opponents first -> then check own suicide -> then check ko.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Optional

SIZE = 9
EMPTY = 0
BLACK = 1
WHITE = 2
KOMI = 2.5
_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class GoGame:
    def __init__(self) -> None:
        self.new_game()

    def new_game(self) -> None:
        self.board: list[int] = [EMPTY] * (SIZE * SIZE)
        self.to_move: int = BLACK
        self.prev_position: Optional[tuple[int, ...]] = None
        self.captures: dict[int, int] = {BLACK: 0, WHITE: 0}
        self.last_move: object = None
        self.finished: bool = False
        self.loser: Optional[int] = None

    # ---------- public API ----------

    def place_stone(self, row: int, col: int) -> bool:
        if self.finished:
            return False
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            return False
        if self.board[row * SIZE + col] != EMPTY:
            return False

        pre_move = tuple(self.board)
        me = self.to_move
        opp = _other(me)
        i = row * SIZE + col
        self.board[i] = me

        captured = 0
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                continue
            if self.board[nr * SIZE + nc] != opp:
                continue
            stones, liberties = self._group(nr, nc)
            if not liberties:
                self._remove(stones)
                captured += len(stones)

        own_stones, own_liberties = self._group(row, col)
        if not own_liberties:
            self.board = list(pre_move)
            return False

        if tuple(self.board) == self.prev_position:
            self.board = list(pre_move)
            return False

        self.prev_position = pre_move
        self.captures[me] += captured
        self.last_move = (row, col)
        self.to_move = opp
        return True

    def is_legal(self, row: int, col: int) -> bool:
        if self.finished:
            return False
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            return False
        if self.board[row * SIZE + col] != EMPTY:
            return False
        probe = copy.deepcopy(self)
        return probe.place_stone(row, col)

    def get_board(self) -> list[list[int]]:
        return [list(self.board[r * SIZE:(r + 1) * SIZE]) for r in range(SIZE)]

    def pass_turn(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.loser = self.to_move
        self.last_move = "pass"

    def end_game(self) -> None:
        self.pass_turn()

    def score(self) -> dict:
        black_stones = sum(1 for v in self.board if v == BLACK)
        white_stones = sum(1 for v in self.board if v == WHITE)
        black_territory, white_territory = self._territory()

        black_total = black_stones + black_territory
        white_total = white_stones + white_territory + KOMI

        if self.loser is not None:
            winner = _other(self.loser)
        else:
            winner = BLACK if black_total > white_total else WHITE

        return {
            "black": black_total,
            "white": white_total,
            "black_stones": black_stones,
            "white_stones": white_stones,
            "black_territory": black_territory,
            "white_territory": white_territory,
            "komi": KOMI,
            "winner": winner,
        }

    # ---------- test-only constructor ----------

    @classmethod
    def from_position(
        cls,
        board: list[list[int]],
        to_move: int = BLACK,
        prev_position: Optional[list[list[int]]] = None,
        captures: Optional[dict[int, int]] = None,
    ) -> "GoGame":
        """TEST ONLY. Inject an arbitrary board state without replaying moves.

        Production callers (MCTS, GUI) must use GoGame() + place_stone()."""
        if len(board) != SIZE or any(len(row) != SIZE for row in board):
            raise ValueError(f"board must be {SIZE}x{SIZE}")
        for row in board:
            for v in row:
                if v not in (EMPTY, BLACK, WHITE):
                    raise ValueError(f"invalid cell value: {v}")
        if to_move not in (BLACK, WHITE):
            raise ValueError(f"invalid to_move: {to_move}")

        g = cls()
        g.board = [board[r][c] for r in range(SIZE) for c in range(SIZE)]
        g.to_move = to_move
        if prev_position is not None:
            if len(prev_position) != SIZE or any(len(row) != SIZE for row in prev_position):
                raise ValueError(f"prev_position must be {SIZE}x{SIZE}")
            g.prev_position = tuple(
                prev_position[r][c] for r in range(SIZE) for c in range(SIZE)
            )
        if captures is not None:
            g.captures = {BLACK: int(captures.get(BLACK, 0)), WHITE: int(captures.get(WHITE, 0))}
        return g

    # ---------- private helpers ----------

    def _group(self, row: int, col: int) -> tuple[set, set]:
        color = self.board[row * SIZE + col]
        stones: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        if color == EMPTY:
            return stones, liberties
        queue = deque([(row, col)])
        stones.add((row, col))
        while queue:
            r, c = queue.popleft()
            for dr, dc in _NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                    continue
                v = self.board[nr * SIZE + nc]
                if v == EMPTY:
                    liberties.add((nr, nc))
                elif v == color and (nr, nc) not in stones:
                    stones.add((nr, nc))
                    queue.append((nr, nc))
        return stones, liberties

    def _remove(self, stones: set) -> None:
        for r, c in stones:
            self.board[r * SIZE + c] = EMPTY

    def _territory(self) -> tuple[int, int]:
        """Chinese area-scoring territory: empty regions bordered by only one color."""
        visited = [False] * (SIZE * SIZE)
        black_t = 0
        white_t = 0
        for start in range(SIZE * SIZE):
            if visited[start] or self.board[start] != EMPTY:
                continue
            region: list[int] = []
            borders: set[int] = set()
            queue = deque([start])
            visited[start] = True
            while queue:
                idx = queue.popleft()
                region.append(idx)
                r, c = divmod(idx, SIZE)
                for dr, dc in _NEIGHBOR_OFFSETS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                        continue
                    nidx = nr * SIZE + nc
                    v = self.board[nidx]
                    if v == EMPTY:
                        if not visited[nidx]:
                            visited[nidx] = True
                            queue.append(nidx)
                    else:
                        borders.add(v)
            if borders == {BLACK}:
                black_t += len(region)
            elif borders == {WHITE}:
                white_t += len(region)
        return black_t, white_t
