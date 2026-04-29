"""
Pure MCTS for 9x9 Go.

Strategy
--------
Why MCTS instead of alpha-beta? Go's branching factor (~80 at the start of a
9x9 game) and the absence of a good static evaluator make alpha-beta weak.
MCTS sidesteps both: it doesn't need to enumerate the tree fully, and the
"value" of a node is the win rate of random rollouts from it.

The four phases per iteration:
  1. Selection   - from the root, descend the tree by picking the child that
                   maximizes UCB1 = win_rate + c * sqrt(ln(parent.N) / N).
                   Stops at a node that has untried moves OR is terminal.
  2. Expansion   - if the node has untried moves, play one and add the
                   resulting state as a new child.
  3. Simulation  - from the expanded node, play random legal non-eye-filling
                   moves to the end. "End" means a player has no candidate
                   moves OR a hard move cap is hit. Score the position.
  4. Backprop    - walk up the path, incrementing visits on every node and
                   incrementing wins on nodes whose mover won the simulation.

Exploration constant c = sqrt(2): the textbook UCB1 default. People sometimes
tune lower (0.7-1.0) for Go specifically; we keep sqrt(2) so the choice is
easy to defend orally.

Two project-specific quirks:
- Pass = concede in this assignment, so MCTS never passes during search or
  rollouts. The agent only passes when literally no legal non-eye move exists.
- Rollouts call score() directly to avoid triggering the pass-loses override.

The "don't fill your own eyes" heuristic is the single change that turns
random rollouts from useless into informative. Without it, random play
kills its own alive groups by filling eyes and the win rate carries no
signal.

Performance note: candidate_moves() is a cheap filter (empties that aren't
own-eyes); it does NOT call is_legal (which deepcopies). Suicide/ko moves
slip through and are caught downstream by place_stone returning False.
That keeps the inner loop fast - is_legal in the hot path was costing
~10x what the rest of MCTS costs.
"""

from __future__ import annotations

import heapq
import math
import random
from typing import Optional

from engine.go_engine import BLACK, EMPTY, SIZE, WHITE, GoGame

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DEFAULT_C = math.sqrt(2)
_ROLLOUT_MOVE_CAP = 200
_TREE_MIN_KEEP = 12
_ROLLOUT_TOP_WINDOW = 8
_ENDGAME_EMPTY_THRESHOLD = 20
_PROG_WIDEN_K = 2.2
_PROG_WIDEN_ALPHA = 0.55
_RAVE_EQUIV = 120.0


def _is_own_eye(board_flat: list[int], r: int, c: int, color: int) -> bool:
    """True if (r,c) is empty and every in-bounds 4-neighbor is `color`.

    4-neighbor definition only (no diagonal-corner refinement). Good
    enough for 9x9 rollouts and easier to defend orally than the full
    eye-shape rule.
    """
    if board_flat[r * SIZE + c] != EMPTY:
        return False
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < SIZE and 0 <= nc < SIZE):
            continue
        if board_flat[nr * SIZE + nc] != color:
            return False
    return True


def _other_color(color: int) -> int:
    return WHITE if color == BLACK else BLACK


def _move_features(game: GoGame, move: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return cheap tactical features for move ordering.

    Features are (capture_size, escapes_atari, adjacent_stones, center_bonus).
    """
    r, c = move
    me = game.to_move
    opp = _other_color(me)
    capture_size = 0
    escapes_atari = 0
    adjacent_stones = 0

    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < SIZE and 0 <= nc < SIZE):
            continue
        v = game.board[nr * SIZE + nc]
        if v == EMPTY:
            continue
        adjacent_stones += 1
        stones, liberties = game._group(nr, nc)  # private helper, cheap and deterministic
        if v == opp and len(liberties) == 1:
            capture_size += len(stones)
        elif v == me and len(liberties) == 1:
            escapes_atari = 1

    center = (SIZE - 1) / 2
    center_bonus = int(8 - (abs(r - center) + abs(c - center)))
    return capture_size, escapes_atari, adjacent_stones, center_bonus


def _move_priority(game: GoGame, move: tuple[int, int]) -> int:
    cap, escape, adj, center = _move_features(game, move)
    # Captures dominate, then urgent defense, then local play.
    priority = 100 * cap + 35 * escape + 8 * adj + center
    if game.board.count(EMPTY) <= _ENDGAME_EMPTY_THRESHOLD:
        # In late game, de-prioritize quiet center-ish plays.
        if cap == 0 and escape == 0 and adj <= 1:
            priority -= 18
    return priority


def _own_territory_indices(board: list[int], color: int) -> set[int]:
    """Flat indices of empty squares lying in regions bordered only by `color`.

    Mirrors GoGame._territory's Chinese-area-scoring definition: an empty
    region that touches only `color` stones (and never the opponent) belongs
    to that color. Filling those points adds no area-score and just wastes a
    tempo, so candidate_moves excludes them. This is a strict superset of
    the single-point own-eye check.
    """
    visited = [False] * (SIZE * SIZE)
    out: set[int] = set()
    for start in range(SIZE * SIZE):
        if visited[start] or board[start] != EMPTY:
            continue
        region: list[int] = []
        borders: set[int] = set()
        stack = [start]
        visited[start] = True
        while stack:
            idx = stack.pop()
            region.append(idx)
            r, c = divmod(idx, SIZE)
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < SIZE and 0 <= nc < SIZE):
                    continue
                nidx = nr * SIZE + nc
                v = board[nidx]
                if v == EMPTY:
                    if not visited[nidx]:
                        visited[nidx] = True
                        stack.append(nidx)
                else:
                    borders.add(v)
        if borders == {color}:
            out.update(region)
    return out


def candidate_moves(game: GoGame) -> list[tuple[int, int]]:
    """Empty points that aren't own-eyes or inside own territory.

    Excluding own territory keeps the bot from wasting moves filling its
    own area: those squares are already counted under Chinese scoring, and
    by definition own-territory regions have no enemy contact, so no
    capture or escape can come from playing inside one. Suicide/ko moves
    may still slip through and are filtered downstream by place_stone
    returning False.
    """
    color = game.to_move
    board = game.board
    own_territory = _own_territory_indices(board, color)
    out: list[tuple[int, int]] = []
    for r in range(SIZE):
        for c in range(SIZE):
            idx = r * SIZE + c
            if board[idx] != EMPTY:
                continue
            if idx in own_territory:
                continue
            if _is_own_eye(board, r, c, color):
                continue
            out.append((r, c))
    return out


def tree_candidate_moves(game: GoGame) -> list[tuple[int, int]]:
    """Priority-ordered tactical candidate set for tree expansion."""
    cands = candidate_moves(game)
    return sorted(cands, key=lambda m: _move_priority(game, m), reverse=True)


def tactical_override_move(game: GoGame) -> Optional[tuple[int, int]]:
    """Conservative tactical override: prefer urgent captures/saves."""
    cands = candidate_moves(game)
    if not cands:
        return None
    ranked = sorted(cands, key=lambda m: _move_priority(game, m), reverse=True)
    for move in ranked[:10]:
        cap, escape, _, _ = _move_features(game, move)
        if cap >= 1 or escape >= 1:
            probe = game.clone_fast()
            if probe.place_stone(*move):
                return move
    return None


class Node:
    __slots__ = (
        "parent", "move", "to_move", "children",
        "ordered_moves", "next_untried_idx", "visits", "wins", "is_terminal",
        "rave_wins", "rave_visits",
    )

    def __init__(
        self,
        parent: Optional["Node"],
        move: Optional[tuple[int, int]],
        game: GoGame,
    ) -> None:
        self.parent = parent
        self.move = move
        self.to_move = game.to_move
        self.children: dict[tuple[int, int], "Node"] = {}
        self.ordered_moves: list[tuple[int, int]] = tree_candidate_moves(game)
        self.next_untried_idx = 0
        self.visits = 0
        self.wins = 0.0
        self.is_terminal = len(self.ordered_moves) == 0
        self.rave_wins: dict[tuple[int, int], float] = {}
        self.rave_visits: dict[tuple[int, int], int] = {}

    def is_fully_expanded(self) -> bool:
        if self.is_terminal:
            return True
        return len(self.children) >= self._expansion_limit()

    def _expansion_limit(self) -> int:
        if self.visits == 0:
            return _TREE_MIN_KEEP
        widened = int(_PROG_WIDEN_K * (self.visits ** _PROG_WIDEN_ALPHA))
        return max(_TREE_MIN_KEEP, widened)

    def best_child_uct(self, c: float) -> "Node":
        log_n = math.log(self.visits)
        best: Optional[Node] = None
        best_score = -math.inf
        for move, child in self.children.items():
            q = child.wins / child.visits
            r_n = self.rave_visits.get(move, 0)
            if r_n:
                r_q = self.rave_wins.get(move, 0.0) / r_n
                beta = _RAVE_EQUIV / (_RAVE_EQUIV + child.visits)
                q = (1.0 - beta) * q + beta * r_q
            score = q + c * math.sqrt(log_n / child.visits)
            if score > best_score:
                best_score = score
                best = child
        assert best is not None
        return best


def _select(root: Node, game: GoGame, c: float) -> tuple[Node, GoGame, list[tuple[int, int]]]:
    node = root
    path_moves: list[tuple[int, int]] = []
    while not node.is_terminal and node.is_fully_expanded() and node.children:
        node = node.best_child_uct(c)
        mover = game.to_move
        if game.place_stone(*node.move):
            path_moves.append((mover, node.move))
    return node, game, path_moves


def _expand(node: Node, game: GoGame, rng: random.Random) -> tuple[Node, GoGame, list[tuple[int, int]]]:
    """Expand one legal child if progressive widening allows it."""
    if node.is_terminal:
        return node, game, []
    if node.is_fully_expanded():
        return node, game, []
    expanded: list[tuple[int, int]] = []
    while node.next_untried_idx < len(node.ordered_moves):
        top = min(len(node.ordered_moves), node.next_untried_idx + 6)
        idx = rng.randrange(node.next_untried_idx, top)
        move = node.ordered_moves[idx]
        node.ordered_moves[idx], node.ordered_moves[node.next_untried_idx] = (
            node.ordered_moves[node.next_untried_idx], node.ordered_moves[idx]
        )
        node.next_untried_idx += 1
        if move in node.children:
            continue
        mover = game.to_move
        if game.place_stone(*move):
            child = Node(parent=node, move=move, game=game)
            node.children[move] = child
            expanded.append((mover, move))
            return child, game, expanded
    if not node.children:
        node.is_terminal = True
    return node, game, expanded


def _simulate(game: GoGame, rng: random.Random) -> tuple[int, list[tuple[int, int]]]:
    """Heuristic-biased rollout to a scored end. Returns winner and played moves."""
    played: list[tuple[int, int]] = []
    for _ in range(_ROLLOUT_MOVE_CAP):
        cands = candidate_moves(game)
        if not cands:
            break
        top_window = _ROLLOUT_TOP_WINDOW
        if game.board.count(EMPTY) <= _ENDGAME_EMPTY_THRESHOLD:
            top_window = 5
        if len(cands) <= top_window:
            window = cands
            weighted = [(move, 1.0 + max(0, _move_priority(game, move)) / 100.0) for move in window]
        else:
            scored = [(_move_priority(game, move), move) for move in cands]
            top = heapq.nlargest(top_window, scored, key=lambda t: t[0])
            weighted = [(move, 1.0 + max(0, score) / 100.0) for score, move in top]
            window = [move for _, move in top]
        # Prefer tactical/local moves but keep randomness for exploration.
        tried = set()
        for _ in range(len(window)):
            moves, weights = zip(*[(m, w) for (m, w) in weighted if m not in tried])
            move = rng.choices(moves, weights=weights, k=1)[0]
            tried.add(move)
            mover = game.to_move
            if game.place_stone(*move):
                played.append((mover, move))
                break
        else:
            break  # every candidate was illegal (rare)
    return game.score()["winner"], played


def _backprop(
    node: Optional[Node],
    winner: int,
    played_moves: list[tuple[int, int]],
) -> None:
    moves_by_player = {BLACK: set(), WHITE: set()}
    for player, move in played_moves:
        moves_by_player[player].add(move)

    while node is not None:
        node.visits += 1
        # node was reached via a move chosen by node.parent.to_move.
        # Increment wins on nodes whose mover (parent.to_move) won.
        if node.parent is not None and node.parent.to_move == winner:
            node.wins += 1.0
        for move in moves_by_player[node.to_move]:
            node.rave_visits[move] = node.rave_visits.get(move, 0) + 1
            if winner == node.to_move:
                node.rave_wins[move] = node.rave_wins.get(move, 0.0) + 1.0
        node = node.parent


def search(
    root_game: GoGame,
    root: Node,
    iterations: int,
    c: float,
    rng: random.Random,
) -> None:
    """Run `iterations` MCTS iterations from `root` (mutates root in place)."""
    for _ in range(iterations):
        game = root_game.clone_fast()
        node, game, path_moves = _select(root, game, c)
        node, game, expanded_moves = _expand(node, game, rng)
        winner, rollout_moves = _simulate(game, rng)
        _backprop(node, winner, path_moves + expanded_moves + rollout_moves)


def best_move(root: Node) -> Optional[tuple[int, int]]:
    """Most-visited child's move. Robust final-move policy
    (less noisy than highest-win-rate). None if root has no children."""
    if not root.children:
        return None
    return max(root.children.items(), key=lambda kv: kv[1].visits)[0]
