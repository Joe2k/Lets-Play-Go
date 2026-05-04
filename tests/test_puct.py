import pytest

from engine.go_engine import BLACK, SIZE, WHITE, GoGame

torch = pytest.importorskip("torch")

from ai.puct_agent import (
    PUCTAgent,
    PUCTNode,
    _apply_tactical_boost,
    _expand_node_from_data,
    run_puct_search,
)


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


def _setup_atari_escape():
    """Set up a position where Black has a stone in atari and must escape.

    Black stone at (4,4) with White stones at (3,4), (5,4), (4,3).
    Only liberty is (4,5). Black to play.
    """
    g = GoGame()
    g.place_stone(4, 4)  # Black
    g.place_stone(3, 4)  # White
    g.place_stone(0, 0)  # Black (filler, far away)
    g.place_stone(5, 4)  # White
    g.place_stone(0, 1)  # Black (filler)
    g.place_stone(4, 3)  # White
    # Now Black at (4,4) has only 1 liberty at (4,5)
    assert g.board[4 * SIZE + 4] == BLACK
    assert g.to_move == BLACK
    return g


def test_tactical_boost_increases_escape_prior():
    """Tactical boost should increase the prior of the escape move."""
    g = _setup_atari_escape()
    node = PUCTNode(to_play=g.to_move, prior=1.0)
    # Uniform network output
    probs = [1.0 / 81.0] * 81
    _expand_node_from_data(node, g, probs, tactical_boost=0.0)
    prior_no_boost = node.children.get((4, 5), PUCTNode(to_play=WHITE, prior=0.0)).prior

    node2 = PUCTNode(to_play=g.to_move, prior=1.0)
    _expand_node_from_data(node2, g, probs, tactical_boost=5.0)
    prior_with_boost = node2.children.get((4, 5), PUCTNode(to_play=WHITE, prior=0.0)).prior

    assert prior_with_boost > prior_no_boost


def test_tactical_boost_increases_capture_prior():
    """Tactical boost should increase the prior of a capture move."""
    g = GoGame()
    # White stone at (4,4) surrounded by Black on 3 sides
    g.place_stone(3, 4)  # Black
    g.place_stone(4, 4)  # White
    g.place_stone(5, 4)  # Black
    g.place_stone(0, 0)  # White (filler)
    g.place_stone(4, 3)  # Black
    g.place_stone(0, 1)  # White (filler)
    # White at (4,4) has only 1 liberty at (4,5). Black to play can capture.
    assert g.to_move == BLACK

    node = PUCTNode(to_play=g.to_move, prior=1.0)
    probs = [1.0 / 81.0] * 81
    _expand_node_from_data(node, g, probs, tactical_boost=0.0)
    prior_no_boost = node.children.get((4, 5), PUCTNode(to_play=WHITE, prior=0.0)).prior

    node2 = PUCTNode(to_play=g.to_move, prior=1.0)
    _expand_node_from_data(node2, g, probs, tactical_boost=5.0)
    prior_with_boost = node2.children.get((4, 5), PUCTNode(to_play=WHITE, prior=0.0)).prior

    assert prior_with_boost > prior_no_boost


def test_tactical_boost_defends_atari_with_search():
    """PUCT with tactical boost should find the escape move when in atari."""
    g = _setup_atari_escape()

    # Verify setup: Black stone at (4,4) should have exactly 1 liberty
    stones, liberties = g._group(4, 4)
    assert len(stones) == 1
    assert len(liberties) == 1
    assert (4, 5) in liberties

    # Without boost: uniform network gives no preference
    root_no = run_puct_search(
        game=g,
        predictor=_UniformPredictor(),
        iterations=400,
        c_puct=1.4,
        tactical_boost=0.0,
    )

    # With boost: escape move should get significantly more visits
    root_with = run_puct_search(
        game=g,
        predictor=_UniformPredictor(),
        iterations=400,
        c_puct=1.4,
        tactical_boost=5.0,
    )

    escape_visits_no = root_no.children.get((4, 5), PUCTNode(to_play=WHITE)).visit_count
    escape_visits_with = root_with.children.get((4, 5), PUCTNode(to_play=WHITE)).visit_count

    # Boost should increase visits to the escape move
    assert escape_visits_with >= escape_visits_no
    # The escape move should have a higher prior with boost
    assert root_with.children[(4, 5)].prior > root_no.children[(4, 5)].prior


def test_tactical_boost_no_effect_on_quiet_positions():
    """In a quiet position with no atari, boost should not change priors."""
    g = GoGame()
    # Just a few stones placed, no atari threats
    g.place_stone(4, 4)
    g.place_stone(4, 5)

    node = PUCTNode(to_play=g.to_move, prior=1.0)
    probs = [1.0 / 81.0] * 81
    _expand_node_from_data(node, g, probs, tactical_boost=0.0)
    priors_no_boost = {m: c.prior for m, c in node.children.items()}

    node2 = PUCTNode(to_play=g.to_move, prior=1.0)
    _expand_node_from_data(node2, g, probs, tactical_boost=5.0)
    priors_with_boost = {m: c.prior for m, c in node2.children.items()}

    # Priors should be nearly identical (no tactical moves to boost)
    for move in priors_no_boost:
        assert abs(priors_no_boost[move] - priors_with_boost[move]) < 1e-6
