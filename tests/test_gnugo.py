import shutil

import pytest

from engine.go_engine import GoGame, SIZE
from ai.gnugo_agent import GnuGoAgent

pytestmark = pytest.mark.skipif(
    shutil.which("gnugo") is None,
    reason="GNU Go binary not installed",
)


def test_gnugo_returns_legal_first_move():
    g = GoGame()
    a = GnuGoAgent(level=1)
    mv = a.select_move(g)
    assert isinstance(mv, tuple)
    r, c = mv
    assert 0 <= r < SIZE and 0 <= c < SIZE
    assert g.is_legal(r, c)


def test_gnugo_syncs_from_history_after_external_move():
    g = GoGame()
    g.place_stone(4, 4)  # Black tengen — applied without the agent seeing it
    a = GnuGoAgent(level=1)
    mv = a.select_move(g)
    assert isinstance(mv, tuple), "GNU Go should reply with a move, not pass"
    r, c = mv
    assert g.is_legal(r, c)
    # Sanity: GNU Go should not try to play on the stone we just placed.
    assert (r, c) != (4, 4)
