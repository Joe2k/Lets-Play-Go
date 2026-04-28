import pytest

from engine.go_engine import GoGame, SIZE

torch = pytest.importorskip("torch")

from ai.puct_agent import PUCTAgent


def test_puct_selects_legal_move_on_empty_board():
    g = GoGame()
    a = PUCTAgent(iterations=30, seed=0)
    move = a.select_move(g)
    assert isinstance(move, tuple)
    r, c = move
    assert 0 <= r < SIZE and 0 <= c < SIZE
    assert g.is_legal(r, c)
