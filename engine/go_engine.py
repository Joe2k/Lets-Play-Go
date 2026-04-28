"""
9x9 Go rules engine.

Board is stored as a flat list of 81 ints (EMPTY/BLACK/WHITE).

Rule order inside place_stone matters:
  capture opponents first -> then check own suicide -> then check ko.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _GroupCache:
    color: int
    stones: set[int]
    liberties: set[int]


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
        self._gid_by_idx: list[int] = [-1] * (SIZE * SIZE)
        self._groups: dict[int, _GroupCache] = {}
        self._next_gid: int = 0
        self._groups_dirty: bool = False

    def clone_fast(self) -> "GoGame":
        """Cheap clone for MCTS hot paths.

        Equivalent to deepcopy for current fields, but avoids generic object
        traversal overhead.
        """
        g = GoGame.__new__(GoGame)
        g.board = self.board.copy()
        g.to_move = self.to_move
        g.prev_position = self.prev_position
        g.captures = self.captures.copy()
        g.last_move = self.last_move
        g.finished = self.finished
        g.loser = self.loser
        g._gid_by_idx = self._gid_by_idx.copy()
        g._groups = {
            gid: _GroupCache(gr.color, gr.stones.copy(), gr.liberties.copy())
            for gid, gr in self._groups.items()
        }
        g._next_gid = self._next_gid
        g._groups_dirty = self._groups_dirty
        return g

    # ---------- public API ----------

    def place_stone(self, row: int, col: int) -> bool:
        if self.finished:
            return False
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            return False
        if self.board[row * SIZE + col] != EMPTY:
            return False

        self._ensure_groups()
        pre_move = tuple(self.board)
        me = self.to_move
        opp = _other(me)
        i = row * SIZE + col
        self.board[i] = me
        friendly_gids: set[int] = set()
        enemy_gids: set[int] = set()
        liberties: set[int] = set()
        for nidx in self._neighbors_of(i):
            v = self.board[nidx]
            if v == EMPTY:
                liberties.add(nidx)
                continue
            gid = self._gid_by_idx[nidx]
            if gid == -1:
                continue
            if v == me:
                friendly_gids.add(gid)
            elif v == opp:
                enemy_gids.add(gid)

        new_gid = self._next_gid
        self._next_gid += 1
        self._groups[new_gid] = _GroupCache(color=me, stones={i}, liberties=liberties)
        self._gid_by_idx[i] = new_gid

        for gid in tuple(friendly_gids):
            if gid not in self._groups:
                continue
            grp = self._groups[gid]
            self._groups[new_gid].stones.update(grp.stones)
            self._groups[new_gid].liberties.update(grp.liberties)
            for sidx in grp.stones:
                self._gid_by_idx[sidx] = new_gid
            del self._groups[gid]
        self._groups[new_gid].liberties.discard(i)

        for gid in enemy_gids:
            if gid in self._groups:
                self._groups[gid].liberties.discard(i)

        captured = 0
        dead_enemy = [gid for gid in enemy_gids if gid in self._groups and not self._groups[gid].liberties]
        for gid in dead_enemy:
            grp = self._groups[gid]
            captured += len(grp.stones)
            for sidx in grp.stones:
                self.board[sidx] = EMPTY
                self._gid_by_idx[sidx] = -1
                for nidx in self._neighbors_of(sidx):
                    ngid = self._gid_by_idx[nidx]
                    if ngid != -1 and ngid in self._groups:
                        self._groups[ngid].liberties.add(sidx)
            del self._groups[gid]

        if not self._groups[new_gid].liberties:
            self.board = list(pre_move)
            self._groups_dirty = True
            self._ensure_groups()
            return False

        if tuple(self.board) == self.prev_position:
            self.board = list(pre_move)
            self._groups_dirty = True
            self._ensure_groups()
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
        probe = self.clone_fast()
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
        neutral_points = self.board.count(EMPTY) - black_territory - white_territory

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
            "neutral_points": neutral_points,
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
        g._groups_dirty = True
        g.to_move = to_move
        if prev_position is not None:
            if len(prev_position) != SIZE or any(len(row) != SIZE for row in prev_position):
                raise ValueError(f"prev_position must be {SIZE}x{SIZE}")
            g.prev_position = tuple(
                prev_position[r][c] for r in range(SIZE) for c in range(SIZE)
            )
        if captures is not None:
            g.captures = {BLACK: int(captures.get(BLACK, 0)), WHITE: int(captures.get(WHITE, 0))}
        g._ensure_groups()
        return g

    # ---------- private helpers ----------

    def _group(self, row: int, col: int) -> tuple[set, set]:
        self._ensure_groups()
        color = self.board[row * SIZE + col]
        stones: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        if color == EMPTY:
            return stones, liberties
        gid = self._gid_by_idx[row * SIZE + col]
        if gid == -1:
            return stones, liberties
        gr = self._groups[gid]
        stones = {divmod(idx, SIZE) for idx in gr.stones}
        liberties = {divmod(idx, SIZE) for idx in gr.liberties}
        return stones, liberties

    def _remove(self, stones: set) -> None:
        for r, c in stones:
            self.board[r * SIZE + c] = EMPTY
        self._groups_dirty = True

    def _neighbors_of(self, idx: int) -> list[int]:
        r, c = divmod(idx, SIZE)
        out: list[int] = []
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE:
                out.append(nr * SIZE + nc)
        return out

    def _ensure_groups(self) -> None:
        if not self._groups_dirty:
            return
        self._rebuild_groups()

    def _rebuild_groups(self) -> None:
        self._gid_by_idx = [-1] * (SIZE * SIZE)
        self._groups = {}
        self._next_gid = 0
        visited = [False] * (SIZE * SIZE)

        for start in range(SIZE * SIZE):
            color = self.board[start]
            if color == EMPTY or visited[start]:
                continue

            gid = self._next_gid
            self._next_gid += 1
            stones: set[int] = set()
            liberties: set[int] = set()
            queue = deque([start])
            visited[start] = True

            while queue:
                idx = queue.popleft()
                stones.add(idx)
                self._gid_by_idx[idx] = gid
                r, c = divmod(idx, SIZE)
                for dr, dc in _NEIGHBOR_OFFSETS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                        continue
                    nidx = nr * SIZE + nc
                    v = self.board[nidx]
                    if v == EMPTY:
                        liberties.add(nidx)
                    elif v == color and not visited[nidx]:
                        visited[nidx] = True
                        queue.append(nidx)

            self._groups[gid] = _GroupCache(color=color, stones=stones, liberties=liberties)

        self._groups_dirty = False

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
