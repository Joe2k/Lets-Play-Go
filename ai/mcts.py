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

import copy
import math
import random
from typing import Optional

from engine.go_engine import EMPTY, SIZE, GoGame

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DEFAULT_C = math.sqrt(2)
_ROLLOUT_MOVE_CAP = 200


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


def candidate_moves(game: GoGame) -> list[tuple[int, int]]:
    """Empty points that aren't own-eyes. May include suicide/ko moves;
    those are filtered downstream by place_stone returning False."""
    color = game.to_move
    board = game.board
    out: list[tuple[int, int]] = []
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r * SIZE + c] != EMPTY:
                continue
            if _is_own_eye(board, r, c, color):
                continue
            out.append((r, c))
    return out


class Node:
    __slots__ = (
        "parent", "move", "to_move", "children",
        "untried_moves", "visits", "wins", "is_terminal",
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
        self.untried_moves: list[tuple[int, int]] = candidate_moves(game)
        self.visits = 0
        self.wins = 0.0
        self.is_terminal = len(self.untried_moves) == 0

    def is_fully_expanded(self) -> bool:
        return not self.untried_moves

    def best_child_uct(self, c: float) -> "Node":
        log_n = math.log(self.visits)
        best: Optional[Node] = None
        best_score = -math.inf
        for child in self.children.values():
            score = (child.wins / child.visits) + c * math.sqrt(log_n / child.visits)
            if score > best_score:
                best_score = score
                best = child
        assert best is not None
        return best


def _select(root: Node, game: GoGame, c: float) -> tuple[Node, GoGame]:
    node = root
    while not node.is_terminal and node.is_fully_expanded() and node.children:
        node = node.best_child_uct(c)
        game.place_stone(*node.move)
    return node, game


def _expand(node: Node, game: GoGame, rng: random.Random) -> tuple[Node, GoGame]:
    """Try untried moves until one is legal; create a child for it. If
    every untried move turns out illegal (suicide/ko), mark the node
    terminal and return it unchanged."""
    if node.is_terminal:
        return node, game
    while node.untried_moves:
        idx = rng.randrange(len(node.untried_moves))
        move = node.untried_moves.pop(idx)
        if game.place_stone(*move):
            child = Node(parent=node, move=move, game=game)
            node.children[move] = child
            return child, game
    if not node.children:
        node.is_terminal = True
    return node, game


def _simulate(game: GoGame, rng: random.Random) -> int:
    """Random rollout to a scored end. Returns BLACK or WHITE."""
    for _ in range(_ROLLOUT_MOVE_CAP):
        cands = candidate_moves(game)
        if not cands:
            break
        rng.shuffle(cands)
        for move in cands:
            if game.place_stone(*move):
                break
        else:
            break  # every candidate was illegal (rare)
    return game.score()["winner"]


def _backprop(node: Optional[Node], winner: int) -> None:
    while node is not None:
        node.visits += 1
        # node was reached via a move chosen by node.parent.to_move.
        # Increment wins on nodes whose mover (parent.to_move) won.
        if node.parent is not None and node.parent.to_move == winner:
            node.wins += 1.0
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
        game = copy.deepcopy(root_game)
        node, game = _select(root, game, c)
        node, game = _expand(node, game, rng)
        winner = _simulate(game, rng)
        _backprop(node, winner)


def best_move(root: Node) -> Optional[tuple[int, int]]:
    """Most-visited child's move. Robust final-move policy
    (less noisy than highest-win-rate). None if root has no children."""
    if not root.children:
        return None
    return max(root.children.items(), key=lambda kv: kv[1].visits)[0]
