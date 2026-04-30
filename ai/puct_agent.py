"""CNN-guided PUCT agent for 9x9 Go."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Protocol, Union

from engine.go_engine import BLACK, SIZE, WHITE, GoGame

from .mcts import candidate_moves
from .model import PASS_INDEX, PolicyValueModel, require_torch

Move = Union[tuple[int, int], str]
PASS_MOVE: str = "pass"

_DIRICHLET_ALPHA = 0.03
_DIRICHLET_WEIGHT = 0.25


@dataclass
class PUCTNode:
    to_play: int
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[Move, "PUCTNode"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class Predictor(Protocol):
    def predict(self, game: GoGame) -> tuple[list[float], float]:
        ...


def _select_child(node: PUCTNode, c_puct: float) -> tuple[Move, PUCTNode]:
    parent_sqrt = math.sqrt(max(1, node.visit_count))
    
    # First Play Urgency (FPU): Assign unvisited nodes the parent's Q value
    # instead of 0.0. This prevents the search from wasting iterations on
    # terrible moves when the parent position is already favorable.
    fpu_value = node.q if node.visit_count > 0 else 0.0
    
    best_score = -1e18
    best_move: Optional[Move] = None
    best_child: Optional[PUCTNode] = None
    for move, child in node.children.items():
        # child.q is from child.to_play's perspective; parent wants the child
        # whose position is worst for the child (i.e., best for parent).
        u = c_puct * child.prior * parent_sqrt / (1 + child.visit_count)
        
        child_q = -child.q if child.visit_count > 0 else fpu_value
        score = child_q + u
        
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    assert best_move is not None and best_child is not None
    return best_move, best_child


def _try_apply_move(sim: GoGame, move: Move) -> bool:
    """Apply `move` to `sim`. Returns whether the move was legal."""
    if move == PASS_MOVE:
        sim.pass_turn()
        return True
    return sim.place_stone(*move)


def _expand_node(node: PUCTNode, game: GoGame, predictor: Predictor) -> float:
    probs, value = predictor.predict(game)
    _expand_node_from_data(node, game, probs)
    return value


def _expand_node_from_data(node: PUCTNode, game: GoGame, probs: list[float]) -> None:
    legal_moves = candidate_moves(game)

    priors: list[tuple[Move, float]] = []
    total = 0.0
    for r, c in legal_moves:
        p = probs[r * SIZE + c]
        priors.append(((r, c), p))
        total += p
    # Pass is always a legal action.
    pass_prior = probs[PASS_INDEX] if len(probs) > PASS_INDEX else 0.0
    priors.append((PASS_MOVE, pass_prior))
    total += pass_prior

    if total <= 1e-12:
        total = float(len(priors))
        priors = [(m, 1.0) for m, _ in priors]
    node.children = {
        move: PUCTNode(to_play=_other(node.to_play), prior=(p / total))
        for move, p in priors
    }
    node.expanded = True


def _add_dirichlet_noise(node: PUCTNode, rng: random.Random) -> None:
    if not node.children:
        return
    moves = list(node.children.keys())
    samples = [rng.gammavariate(_DIRICHLET_ALPHA, 1.0) for _ in moves]
    s = sum(samples)
    if s <= 0:
        return
    noise = [x / s for x in samples]
    for move, n in zip(moves, noise):
        ch = node.children[move]
        ch.prior = (1.0 - _DIRICHLET_WEIGHT) * ch.prior + _DIRICHLET_WEIGHT * n


_MAX_DESCENT_DEPTH = 60


def run_puct_search(
    game: GoGame,
    predictor: Predictor,
    iterations: int,
    c_puct: float,
    root: Optional[PUCTNode] = None,
    add_root_noise: bool = False,
    rng: Optional[random.Random] = None,
    max_descent_depth: int = _MAX_DESCENT_DEPTH,
) -> PUCTNode:
    if root is None:
        root = PUCTNode(to_play=game.to_move, prior=1.0)
    if not root.expanded:
        _expand_node(root, game, predictor)
    if add_root_noise:
        _add_dirichlet_noise(root, rng or random.Random())

    for _ in range(iterations):
        sim = game.clone_fast()
        node = root
        path = [node]

        # Selection: descend while the current node has expanded children.
        while node.expanded and node.children and len(path) <= max_descent_depth:
            move, child = _select_child(node, c_puct=c_puct)
            if not _try_apply_move(sim, move):
                del node.children[move]
                if not node.children:
                    break
                continue
            node = child
            path.append(node)
            if sim.finished:
                break

        # Evaluate leaf
        if sim.finished:
            winner = sim.score()["winner"]
            leaf_value = 1.0 if winner == node.to_play else -1.0
        elif not node.expanded:
            leaf_value = _expand_node(node, sim, predictor)
        elif not node.children:
            leaf_value = -1.0
        else:
            leaf_value = 0.0

        # Backprop
        v = leaf_value
        for p in reversed(path):
            p.visit_count += 1
            p.value_sum += v
            v = -v

    return root


def run_batched_puct_search(
    games: list[GoGame],
    model: PolicyValueModel,
    iterations: int,
    c_puct: float,
    roots: list[PUCTNode],
    add_root_noise: bool = False,
    rng: Optional[random.Random] = None,
    max_descent_depth: int = _MAX_DESCENT_DEPTH,
) -> list[PUCTNode]:
    """Run MCTS iterations for a batch of games in parallel, batching NN evaluations."""
    for i, root in enumerate(roots):
        if not root.expanded:
            _expand_node(root, games[i], model)
        if add_root_noise:
            _add_dirichlet_noise(root, rng or random.Random())

    for _ in range(iterations):
        sims = [g.clone_fast() for g in games]
        paths = [[r] for r in roots]
        leaf_nodes: list[PUCTNode] = []

        for i in range(len(games)):
            sim = sims[i]
            node = roots[i]
            path = paths[i]

            while node.expanded and node.children and len(path) <= max_descent_depth:
                move, child = _select_child(node, c_puct=c_puct)
                if not _try_apply_move(sim, move):
                    del node.children[move]
                    if not node.children:
                        break
                    continue
                node = child
                path.append(node)
                if sim.finished:
                    break
            leaf_nodes.append(node)

        to_predict_indices = []
        to_predict_sims = []
        leaf_values = [0.0] * len(games)

        for i, (node, sim) in enumerate(zip(leaf_nodes, sims)):
            if sim.finished:
                winner = sim.score()["winner"]
                leaf_values[i] = 1.0 if winner == node.to_play else -1.0
            elif not node.expanded:
                to_predict_indices.append(i)
                to_predict_sims.append(sim)
            elif not node.children:
                leaf_values[i] = -1.0
            else:
                leaf_values[i] = 0.0

        if to_predict_sims:
            probs_list, values = model.predict_batch(to_predict_sims)
            for i, idx in enumerate(to_predict_indices):
                node = leaf_nodes[idx]
                sim = to_predict_sims[i]
                _expand_node_from_data(node, sim, probs_list[i])
                leaf_values[idx] = values[i]

        for i in range(len(games)):
            v = leaf_values[i]
            for p in reversed(paths[i]):
                p.visit_count += 1
                p.value_sum += v
                v = -v

    return roots


class PUCTAgent:
    def __init__(
        self,
        iterations: int = 400,
        c_puct: float = 1.4,
        model_path: Optional[str] = None,
        seed: Optional[int] = None,
        device: str = "cpu",
        add_root_noise: bool = False,
    ) -> None:
        require_torch()
        self.iterations = iterations
        self.c_puct = c_puct
        self._rng = random.Random(seed)
        self.model = PolicyValueModel(model_path=model_path, device=device)
        self.add_root_noise = add_root_noise
        self._root: Optional[PUCTNode] = None
        self._last_to_play: Optional[int] = None

    def reset(self) -> None:
        self._root = None
        self._last_to_play = None

    def select_move(self, game: GoGame) -> Move:
        self._advance_root_to(game)

        self._root = run_puct_search(
            game=game,
            predictor=self.model,
            iterations=self.iterations,
            c_puct=self.c_puct,
            root=self._root,
            add_root_noise=self.add_root_noise,
            rng=self._rng,
        )
        if not self._root.children:
            return PASS_MOVE
        best = max(self._root.children.items(), key=lambda kv: kv[1].visit_count)[0]
        # Promote chosen child to next root for tree reuse.
        self._root = self._root.children[best]
        self._last_to_play = _other(game.to_move)
        return best

    def _advance_root_to(self, game: GoGame) -> None:
        """Descend the stored root to mirror the opponent's last move, if known."""
        if self._root is None:
            self._root = PUCTNode(to_play=game.to_move, prior=1.0)
            self._last_to_play = None
            return
        last = game.last_move
        key: Optional[Move] = None
        if isinstance(last, tuple):
            key = last
        elif last == "pass":
            key = PASS_MOVE
        if key is not None and key in self._root.children:
            self._root = self._root.children[key]
            self._last_to_play = game.to_move
            return
        # Mismatch — discard and rebuild fresh.
        self._root = PUCTNode(to_play=game.to_move, prior=1.0)
        self._last_to_play = None
