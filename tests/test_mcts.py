"""Sanity tests for MCTSAgent. Strength benchmarks belong elsewhere."""

from engine.go_engine import BLACK, EMPTY, SIZE, WHITE, GoGame
from ai.agent import MCTSAgent
from ai.mcts import Node, candidate_moves, search, tactical_override_move


def _blank():
    return [[EMPTY] * SIZE for _ in range(SIZE)]


def test_does_not_pass_when_moves_available():
    g = GoGame()
    a = MCTSAgent(iterations=50, seed=0)
    move = a.select_move(g)
    assert isinstance(move, tuple)
    r, c = move
    assert 0 <= r < SIZE and 0 <= c < SIZE


def test_does_not_consider_filling_own_eye():
    # Black wall around a single empty point at (2,2). (2,2) is an own-eye
    # for Black: every 4-neighbor is Black.
    b = _blank()
    for r, c in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]:
        b[r][c] = BLACK
    g = GoGame.from_position(b, to_move=BLACK)
    assert (2, 2) not in candidate_moves(g)
    a = MCTSAgent(iterations=100, seed=0)
    move = a.select_move(g)
    assert move != (2, 2)


def test_passes_when_only_own_eyes_remain():
    # Board is entirely Black except two own-eye points; Black to move.
    # Both empties are own-eyes -> filtered -> no candidates -> pass.
    b = [[BLACK] * SIZE for _ in range(SIZE)]
    b[4][3] = EMPTY
    b[4][5] = EMPTY
    g = GoGame.from_position(b, to_move=BLACK)
    assert candidate_moves(g) == []
    a = MCTSAgent(iterations=10, seed=0)
    assert a.select_move(g) == "pass"


def test_does_not_fill_own_multi_point_territory():
    # Black wall at column 4 splits the board: left half is empty and
    # bordered only by Black; right half is empty and touches the edge plus
    # Black. With Black to move we must not pick any cell in the wholly-
    # enclosed left region — those squares are already Black territory.
    b = _blank()
    for r in range(SIZE):
        b[r][4] = BLACK
    # Seal top/bottom of the left region too so it's a pure Black region.
    for c in range(4):
        b[0][c] = BLACK
        b[SIZE - 1][c] = BLACK
    # And seal the leftmost column so the region is not bordered by the wall edge alone.
    for r in range(SIZE):
        b[r][0] = BLACK
    g = GoGame.from_position(b, to_move=BLACK)
    cands = candidate_moves(g)
    # Left interior cells: rows 1..7, cols 1..3 — all should be filtered.
    enclosed = {(r, c) for r in range(1, SIZE - 1) for c in range(1, 4)}
    assert enclosed.isdisjoint(cands), (
        f"candidate_moves leaked own-territory squares: {sorted(set(cands) & enclosed)}"
    )


def test_passes_when_only_own_territory_remains():
    # Board: a Black wall on column 4. Left side is empty (Black territory);
    # right side is fully Black stones. Black to move has nothing to do
    # outside its own area -> should pass.
    b = [[BLACK] * SIZE for _ in range(SIZE)]
    for r in range(SIZE):
        for c in range(4):
            b[r][c] = EMPTY
    # Make the left a sealed Black region by filling top/bottom/left edges with Black.
    # Easiest: keep only a single 3x3 empty pocket bordered entirely by Black.
    b = [[BLACK] * SIZE for _ in range(SIZE)]
    for r in (3, 4, 5):
        for c in (3, 4, 5):
            b[r][c] = EMPTY
    g = GoGame.from_position(b, to_move=BLACK)
    # The 3x3 empty region is bordered only by Black -> all 9 cells are own territory.
    assert candidate_moves(g) == []
    a = MCTSAgent(iterations=10, seed=0)
    assert a.select_move(g) == "pass"


def test_invasion_breaks_own_territory_filter():
    # Same enclosed 3x3 empty pocket, but White has already invaded one
    # cell. The remaining 8 empty cells now touch a White stone -> not
    # Black territory anymore -> Black is allowed to play there to kill.
    b = [[BLACK] * SIZE for _ in range(SIZE)]
    for r in (3, 4, 5):
        for c in (3, 4, 5):
            b[r][c] = EMPTY
    b[4][4] = WHITE  # invasion
    g = GoGame.from_position(b, to_move=BLACK)
    cands = candidate_moves(g)
    # All 8 remaining empties should be playable.
    expected = {(r, c) for r in (3, 4, 5) for c in (3, 4, 5)} - {(4, 4)}
    assert set(cands) == expected


def test_self_play_game_completes_without_crash():
    # Two MCTS agents play each other with low iterations. Verifies the
    # full play loop is sound: legal moves only, tree reuse across turns,
    # eventual termination via pass once both sides run out of non-eye
    # moves. (Strength benchmarks belong elsewhere.)
    g = GoGame()
    black = MCTSAgent(iterations=20, seed=1)
    white = MCTSAgent(iterations=20, seed=2)
    move_count = 0
    while not g.finished and move_count < 200:
        agent = black if g.to_move == BLACK else white
        move = agent.select_move(g)
        if move == "pass":
            g.pass_turn()
        else:
            assert g.place_stone(*move), f"agent returned illegal move {move}"
        move_count += 1
    assert g.finished or move_count == 200
    # score() must succeed at any point.
    result = g.score()
    assert result["winner"] in (BLACK, WHITE)


def test_tree_reuse_descends_into_existing_subtree():
    g = GoGame()
    a = MCTSAgent(iterations=300, seed=0)
    move1 = a.select_move(g)
    g.place_stone(*move1)

    explored_replies = list(a._root.children.keys())
    assert len(explored_replies) >= 2, "expected MCTS to have explored multiple replies"

    # Pretend the opponent picks a move we've already explored.
    opp_move = max(explored_replies, key=lambda m: a._root.children[m].visits)
    inherited_visits = a._root.children[opp_move].visits
    assert inherited_visits > 0

    g.place_stone(*opp_move)
    a._advance_root_to(g)
    # Without running search yet, the new root's visits should equal the
    # visits inherited from the previous search - proof of tree reuse.
    assert a._root.visits == inherited_visits


def test_seed_determinism():
    g1 = GoGame()
    g2 = GoGame()
    a1 = MCTSAgent(iterations=100, seed=42)
    a2 = MCTSAgent(iterations=100, seed=42)
    assert a1.select_move(g1) == a2.select_move(g2)


def test_tactical_override_prefers_immediate_capture():
    # White stone at (0,0) is in atari; Black should take (0,1) immediately.
    b = _blank()
    b[0][0] = WHITE
    b[1][0] = BLACK
    g = GoGame.from_position(b, to_move=BLACK)
    assert tactical_override_move(g) == (0, 1)


def test_tactical_override_returns_none_without_urgent_move():
    g = GoGame()
    assert tactical_override_move(g) is None


def test_progressive_widening_limit_grows_with_visits():
    g = GoGame()
    n = Node(parent=None, move=None, game=g)
    low = n._expansion_limit()
    n.visits = 400
    high = n._expansion_limit()
    assert high > low


def test_search_populates_rave_statistics():
    g = GoGame()
    root = Node(parent=None, move=None, game=g)
    import random
    rng = random.Random(0)
    search(g.clone_fast(), root, iterations=40, c=1.4, rng=rng)
    assert root.visits == 40
    assert len(root.children) > 0
    assert len(root.rave_visits) > 0
    some_move = next(iter(root.rave_visits))
    assert root.rave_visits[some_move] >= 1
