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
HISTORY_DEPTH = 2  # number of pre-move board snapshots kept for NN input planes
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
        self.history: list[tuple[int, int] | str] = []
        self.finished: bool = False
        self.loser: Optional[int] = None
        # Pre-move board snapshots, ordered most-recent-first, capped at
        # HISTORY_DEPTH. Used by the NN encoder to build history planes;
        # carried by clone_fast so MCTS leaf evaluations see correct history.
        self.prev_boards: list[list[int]] = []
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
        g.history = self.history.copy()
        g.finished = self.finished
        g.loser = self.loser
        g.prev_boards = [b.copy() for b in self.prev_boards]
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
        self.prev_boards.insert(0, list(pre_move))
        if len(self.prev_boards) > HISTORY_DEPTH:
            self.prev_boards = self.prev_boards[:HISTORY_DEPTH]
        self.captures[me] += captured
        self.last_move = (row, col)
        self.history.append((row, col))
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
        """The current player passes. Two consecutive passes ends the game.
        This allows GNU Go to finish naturally and be scored correctly."""
        if self.finished:
            return
        # Pass doesn't change the board, but it advances time — push a
        # snapshot so history planes shift correctly.
        self.prev_boards.insert(0, self.board.copy())
        if len(self.prev_boards) > HISTORY_DEPTH:
            self.prev_boards = self.prev_boards[:HISTORY_DEPTH]
        if self.last_move == "pass":
            self.finished = True
        self.last_move = "pass"
        self.history.append("pass")
        self.to_move = _other(self.to_move)

    def concede(self) -> None:
        """Immediate concession (the old behavior of pass_turn)."""
        if self.finished:
            return
        self.finished = True
        self.loser = self.to_move
        self.last_move = "concede"
        self.history.append("concede")

    def end_game(self) -> None:
        self.pass_turn()

    def score(self) -> dict:
        # Dead-stone detection (Benson's unconditional life + conservative
        # enclosure rule) only kicks in when the game ended via two passes.
        # Concession path is left alone — the loser is forced. Mid-game and
        # not-yet-finished positions use the pure boundary-color rule so the
        # GUI's live score display is unchanged.
        if self.finished and self.loser is None:
            return self.score_final()
        else:
            black_stones = self.board.count(BLACK)
            white_stones = self.board.count(WHITE)
            black_territory, white_territory = self._territory()

        # Reported relative to the (possibly dead-removed) synthetic board so
        # stones + territory + neutral always sums to the full board.
        neutral_points = (
            (SIZE * SIZE) - black_stones - white_stones - black_territory - white_territory
        )

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

    def score_final(self) -> dict:
        """Score using dead-stone detection regardless of game state."""
        self._ensure_groups()
        alive_b = self._benson_alive(BLACK)
        alive_w = self._benson_alive(WHITE)
        dead_b, dead_w = self._dead_chains(alive_b, alive_w)
        black_stones, white_stones, black_territory, white_territory = (
            self._score_with_dead_removed(dead_b | dead_w)
        )

        # Reported relative to the (possibly dead-removed) synthetic board so
        # stones + territory + neutral always sums to the full board.
        neutral_points = (
            (SIZE * SIZE) - black_stones - white_stones - black_territory - white_territory
        )

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

    def ownership_map(self) -> list[float]:
        """Return length-81 ownership from Black's perspective.

        +1.0 = Black owns the point, -1.0 = White owns, 0.0 = neutral/dame.
        Uses dead-stone removal when the game ended via two passes.
        """
        if self.finished and self.loser is None:
            self._ensure_groups()
            alive_b = self._benson_alive(BLACK)
            alive_w = self._benson_alive(WHITE)
            dead_b, dead_w = self._dead_chains(alive_b, alive_w)
            board = list(self.board)
            for s in dead_b | dead_w:
                board[s] = EMPTY
        else:
            board = self.board

        visited = [False] * (SIZE * SIZE)
        owner = [0.0] * (SIZE * SIZE)

        for i in range(SIZE * SIZE):
            if board[i] == BLACK:
                owner[i] = 1.0
            elif board[i] == WHITE:
                owner[i] = -1.0

        for start in range(SIZE * SIZE):
            if visited[start] or board[start] != EMPTY:
                continue
            region: list[int] = []
            borders: set[int] = set()
            queue = deque([start])
            visited[start] = True
            while queue:
                idx = queue.popleft()
                region.append(idx)
                for nidx in self._neighbors_of(idx):
                    v = board[nidx]
                    if v == EMPTY:
                        if not visited[nidx]:
                            visited[nidx] = True
                            queue.append(nidx)
                    else:
                        borders.add(v)
            val = 0.0
            if borders == {BLACK}:
                val = 1.0
            elif borders == {WHITE}:
                val = -1.0
            for idx in region:
                owner[idx] = val
        return owner

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
        g.prev_boards = []
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

    # ---------- dead-stone detection (Benson's unconditional life) ----------
    #
    # We use this only when the game ended via two passes (see score()). The
    # algorithm is conservative: it only kills groups that are not Benson-
    # alive AND are fully enclosed by Benson-alive enemy chains. So it will
    # under-kill in seki and complex shapes but never wrongly remove a group
    # that has any chance to live.

    def _benson_alive(self, color: int) -> set[int]:
        """Group-ids of `color` that are unconditionally alive.

        A chain is unconditionally alive iff it has at least 2 "vital small"
        regions with respect to a candidate set S that itself satisfies the
        condition. We compute the largest such S by iteratively dropping any
        chain that fails the test. See Benson 1976.
        """
        own_gids = [gid for gid, gr in self._groups.items() if gr.color == color]
        if not own_gids:
            return set()

        # 1. Find the color-regions: maximal 4-connected components of
        #    (EMPTY ∪ opposite-color) cells. Track which of `color`'s chains
        #    border each region, and the set of empty cells in the region.
        opp = _other(color)
        visited = [False] * (SIZE * SIZE)
        # region_idx -> (empty_cells, border_gids)
        regions: list[tuple[set[int], set[int]]] = []
        for start in range(SIZE * SIZE):
            if visited[start] or self.board[start] == color:
                continue
            empty_cells: set[int] = set()
            border_gids: set[int] = set()
            queue = deque([start])
            visited[start] = True
            while queue:
                idx = queue.popleft()
                v = self.board[idx]
                if v == EMPTY:
                    empty_cells.add(idx)
                # Whether v is EMPTY or opp, walk neighbors.
                for nidx in self._neighbors_of(idx):
                    nv = self.board[nidx]
                    if nv == color:
                        border_gids.add(self._gid_by_idx[nidx])
                    elif not visited[nidx]:
                        visited[nidx] = True
                        queue.append(nidx)
            regions.append((empty_cells, border_gids))

        # 2. For each (region, chain) pair, precompute "vital": every empty
        #    cell in the region is adjacent to at least one stone of the
        #    chain. This depends only on the position, not on S.
        vital: dict[tuple[int, int], bool] = {}
        for ri, (empty_cells, border_gids) in enumerate(regions):
            for gid in border_gids:
                stones = self._groups[gid].stones
                ok = True
                for ec in empty_cells:
                    if not any(n in stones for n in self._neighbors_of(ec)):
                        ok = False
                        break
                vital[(ri, gid)] = ok

        # 3. Iterate: drop any chain in S without ≥2 regions that are
        #    "small wrt S" (every border_gid in S) AND vital for it.
        S = set(own_gids)
        while True:
            changed = False
            for gid in list(S):
                count = 0
                for ri, (_empty, border_gids) in enumerate(regions):
                    if gid not in border_gids:
                        continue
                    if not border_gids.issubset(S):
                        continue
                    if vital.get((ri, gid), False):
                        count += 1
                        if count >= 2:
                            break
                if count < 2:
                    S.discard(gid)
                    changed = True
            if not changed:
                break
        return S

    def _dead_chains(
        self, alive_black: set[int], alive_white: set[int]
    ) -> tuple[set[int], set[int]]:
        """Conservative dead-stone detection.

        A chain X of color C is dead iff:
          - X is not in alive_C, AND
          - flood-filling from X through EMPTY cells and other own-color
            chains reaches at least one opponent chain, and EVERY opponent
            chain reached is in alive_(-C).

        Returns (dead_black_indices, dead_white_indices) as flat-board
        index sets. Opponent stones terminate the flood (we never cross
        them); they are recorded as boundaries.
        """
        dead_black: set[int] = set()
        dead_white: set[int] = set()
        seen: set[int] = set()
        for gid, gr in self._groups.items():
            if gid in seen:
                continue
            color = gr.color
            alive_us = alive_black if color == BLACK else alive_white
            alive_them = alive_white if color == BLACK else alive_black
            if gid in alive_us:
                continue

            # BFS from this chain through empties and same-color chains.
            visited = set(gr.stones)
            queue = deque(gr.stones)
            visited_chains = {gid}
            opp_chains_touched: set[int] = set()
            while queue:
                idx = queue.popleft()
                for nidx in self._neighbors_of(idx):
                    if nidx in visited:
                        continue
                    nv = self.board[nidx]
                    ngid = self._gid_by_idx[nidx]
                    if nv == EMPTY:
                        visited.add(nidx)
                        queue.append(nidx)
                    elif nv == color:
                        # Skip alive allies — otherwise a dead chain
                        # whose flood reaches a shared empty eye would
                        # drag the alive ally into the dead set too.
                        if ngid in alive_us:
                            continue
                        if ngid not in visited_chains:
                            visited_chains.add(ngid)
                            for s in self._groups[ngid].stones:
                                if s not in visited:
                                    visited.add(s)
                                    queue.append(s)
                    else:
                        # Opponent stone: record boundary, don't cross.
                        opp_chains_touched.add(ngid)

            # Dead iff we touched any opponent and every opponent we touched
            # is unconditionally alive.
            if opp_chains_touched and opp_chains_touched.issubset(alive_them):
                target = dead_black if color == BLACK else dead_white
                for cgid in visited_chains:
                    target.update(self._groups[cgid].stones)
            seen.update(visited_chains)

        # Fallback: mark individual non-alive chains in atari (≤1 liberty)
        # as dead when their liberty cannot be defended by a Benson-alive
        # friendly chain AND the opponent can legally play the capturing move.
        # This catches stones that are obviously capturable but sit inside an
        # enclosure whose captor is not itself unconditionally alive.
        for gid, gr in self._groups.items():
            if gr.color == BLACK:
                if gid in alive_black or gid in dead_black:
                    continue
                target = dead_black
            else:
                if gid in alive_white or gid in dead_white:
                    continue
                target = dead_white

            liberties: set[int] = set()
            for s in gr.stones:
                for nidx in self._neighbors_of(s):
                    if self.board[nidx] == EMPTY:
                        liberties.add(nidx)
            if len(liberties) > 1:
                continue

            if not self._is_capturable(gid):
                continue

            target.update(gr.stones)
            seen.add(gid)

        return dead_black, dead_white

    def _is_capturable(self, chain_gid: int) -> bool:
        """Return True if the opponent can legally capture this chain."""
        gr = self._groups[chain_gid]
        liberties: set[int] = set()
        for s in gr.stones:
            for nidx in self._neighbors_of(s):
                if self.board[nidx] == EMPTY:
                    liberties.add(nidx)
        if not liberties:
            return True
        if len(liberties) != 1:
            return False
        lib = liberties.pop()
        r, c = divmod(lib, SIZE)
        probe = self.clone_fast()
        probe.to_move = _other(gr.color)
        probe.finished = False
        return probe.place_stone(r, c)

    def _score_with_dead_removed(
        self, dead: set[int]
    ) -> tuple[int, int, int, int]:
        """Return (black_stones, white_stones, black_territory, white_territory)
        on a synthetic board where `dead` indices are treated as EMPTY.

        Does NOT mutate self.board or self._groups. Uses the same boundary-
        color flood as _territory().
        """
        if not dead:
            bt, wt = self._territory()
            return self.board.count(BLACK), self.board.count(WHITE), bt, wt

        board = list(self.board)
        for s in dead:
            board[s] = EMPTY
        black_stones = board.count(BLACK)
        white_stones = board.count(WHITE)

        visited = [False] * (SIZE * SIZE)
        black_t = 0
        white_t = 0
        for start in range(SIZE * SIZE):
            if visited[start] or board[start] != EMPTY:
                continue
            region: list[int] = []
            borders: set[int] = set()
            queue = deque([start])
            visited[start] = True
            while queue:
                idx = queue.popleft()
                region.append(idx)
                for nidx in self._neighbors_of(idx):
                    v = board[nidx]
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
        return black_stones, white_stones, black_t, white_t
