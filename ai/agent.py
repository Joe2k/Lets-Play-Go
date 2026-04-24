"""
MCTSAgent: GUI-facing wrapper around the MCTS search.

Owns a persistent search tree and reuses subtrees across moves. When the
opponent plays, we descend our stored root into the matching child and
promote it as the new root, preserving all the visits/wins below it
instead of throwing away the search work.

Pass policy is project-specific: passing concedes in this assignment,
so the agent only returns "pass" if there are literally zero legal
non-eye-filling moves on the actual board.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Optional, Union

from engine.go_engine import GoGame

from .mcts import Node, best_move, candidate_moves, search

_DEFAULT_C = math.sqrt(2)

Move = Union[tuple[int, int], str]  # (r, c) or "pass"


class MCTSAgent:
    def __init__(
        self,
        iterations: int = 2000,
        c: float = _DEFAULT_C,
        seed: Optional[int] = None,
    ) -> None:
        self.iterations = iterations
        self.c = c
        self._rng = random.Random(seed)
        self._root: Optional[Node] = None

    def reset(self) -> None:
        self._root = None

    def select_move(self, game: GoGame) -> Move:
        if not candidate_moves(game):
            self._root = None
            return "pass"

        self._advance_root_to(game)
        search_game = copy.deepcopy(game)
        search(search_game, self._root, self.iterations, self.c, self._rng)

        move = best_move(self._root)
        if move is None:
            self._root = None
            return "pass"

        self._promote(move)
        return move

    def _advance_root_to(self, game: GoGame) -> None:
        """Make self._root represent `game`. Reuses an existing subtree when
        the opponent's last move is one we already explored."""
        if self._root is not None:
            last = game.last_move
            if isinstance(last, tuple) and last in self._root.children:
                self._promote(last)
                if self._root.to_move == game.to_move:
                    return
        self._root = Node(parent=None, move=None, game=game)

    def _promote(self, move: tuple[int, int]) -> None:
        assert self._root is not None
        child = self._root.children[move]
        child.parent = None
        self._root = child
