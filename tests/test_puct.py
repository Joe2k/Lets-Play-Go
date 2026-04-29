import pytest

from engine.go_engine import GoGame, SIZE

torch = pytest.importorskip("torch")

from ai.puct_agent import PUCTAgent, PUCTNode, run_puct_search


def test_puct_selects_legal_move_on_empty_board():
    g = GoGame()
    a = PUCTAgent(iterations=30, seed=0)
    move = a.select_move(g)
    assert isinstance(move, tuple)
    r, c = move
    assert 0 <= r < SIZE and 0 <= c < SIZE
    assert g.is_legal(r, c)


class _UniformPredictor:
    def predict(self, game):
        return [1.0 / 81.0] * 81, 0.0


def test_run_puct_search_respects_max_descent_depth():
    """A pathologically deep persisted tree must not blow up a single search.

    Builds a chain of 500 expanded nodes (much deeper than max_descent_depth=5)
    and confirms run_puct_search returns promptly with the iteration budget
    accounted for.
    """
    g = GoGame()
    root = PUCTNode(to_play=g.to_move, prior=1.0)
    chain = root
    for i in range(500):
        child_move = (i // 9 % 9, i % 9)
        child = PUCTNode(to_play=2 if chain.to_play == 1 else 1, prior=1.0)
        chain.children = {child_move: child}
        chain.expanded = True
        chain = child
    iterations = 8
    out = run_puct_search(
        game=g,
        predictor=_UniformPredictor(),
        iterations=iterations,
        c_puct=1.4,
        root=root,
        max_descent_depth=5,
    )
    assert out is root
    assert root.visit_count == iterations
