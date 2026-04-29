"""Rules-engine tests for GoGame. Each test name targets one rule."""

import copy
import pytest

from engine.go_engine import BLACK, EMPTY, KOMI, SIZE, WHITE, GoGame


def blank_board():
    return [[EMPTY] * SIZE for _ in range(SIZE)]


def test_new_game_empty():
    g = GoGame()
    board = g.get_board()
    assert len(board) == SIZE and all(len(row) == SIZE for row in board)
    assert sum(v for row in board for v in row) == 0
    assert g.to_move == BLACK
    assert g.captures == {BLACK: 0, WHITE: 0}
    assert g.finished is False


def test_simple_placement_and_turn_swap():
    g = GoGame()
    assert g.place_stone(4, 4) is True
    assert g.get_board()[4][4] == BLACK
    assert g.to_move == WHITE
    assert g.place_stone(3, 3) is True
    assert g.get_board()[3][3] == WHITE
    assert g.to_move == BLACK


def test_single_stone_capture():
    b = blank_board()
    b[0][0] = BLACK
    b[1][0] = WHITE
    g = GoGame.from_position(b, to_move=WHITE)
    assert g.place_stone(0, 1) is True
    assert g.get_board()[0][0] == EMPTY
    assert g.get_board()[0][1] == WHITE
    assert g.captures[WHITE] == 1


def test_edge_capture_two_stone_group():
    b = blank_board()
    b[0][0] = BLACK
    b[0][1] = BLACK
    b[0][2] = WHITE
    b[1][0] = WHITE
    g = GoGame.from_position(b, to_move=WHITE)
    assert g.place_stone(1, 1) is True
    assert g.get_board()[0][0] == EMPTY
    assert g.get_board()[0][1] == EMPTY
    assert g.captures[WHITE] == 2


def test_multi_stone_capture_center():
    b = blank_board()
    b[4][4] = BLACK
    b[5][4] = BLACK
    b[3][4] = WHITE
    b[4][3] = WHITE
    b[4][5] = WHITE
    b[5][3] = WHITE
    b[5][5] = WHITE
    g = GoGame.from_position(b, to_move=WHITE)
    assert g.place_stone(6, 4) is True
    assert g.get_board()[4][4] == EMPTY
    assert g.get_board()[5][4] == EMPTY
    assert g.captures[WHITE] == 2


def test_suicide_rejected_single_stone():
    b = blank_board()
    b[0][1] = WHITE
    b[0][2] = WHITE
    b[1][0] = WHITE
    b[2][0] = WHITE
    g = GoGame.from_position(b, to_move=BLACK)
    assert g.is_legal(0, 0) is False
    assert g.place_stone(0, 0) is False
    assert g.get_board()[0][0] == EMPTY


def test_suicide_rejected_multi_stone():
    b = blank_board()
    b[0][1] = BLACK
    b[0][2] = WHITE
    b[1][0] = WHITE
    b[1][1] = WHITE
    g = GoGame.from_position(b, to_move=BLACK)
    assert g.is_legal(0, 0) is False
    assert g.place_stone(0, 0) is False
    assert g.get_board()[0][0] == EMPTY
    assert g.get_board()[0][1] == BLACK


def test_legal_self_atari_that_captures():
    # Black plays (4,4); without capture-first it would be suicide (all 4
    # neighbors White). But the White stone at (3,4) has exactly one liberty
    # (the move point), so it is captured first and Black ends with 1 liberty.
    b = blank_board()
    b[2][4] = BLACK
    b[3][3] = BLACK
    b[3][5] = BLACK
    b[3][4] = WHITE
    b[4][3] = WHITE
    b[4][5] = WHITE
    b[5][4] = WHITE
    g = GoGame.from_position(b, to_move=BLACK)
    assert g.is_legal(4, 4) is True
    assert g.place_stone(4, 4) is True
    assert g.get_board()[4][4] == BLACK
    assert g.get_board()[3][4] == EMPTY
    assert g.captures[BLACK] == 1


def test_ko_basic():
    # Classic 1-stone ko. Black captures a single White stone; White cannot
    # immediately recapture on the same spot.
    b = blank_board()
    b[0][1] = BLACK
    b[1][0] = BLACK
    b[2][1] = BLACK
    b[0][2] = WHITE
    b[1][1] = WHITE
    b[1][3] = WHITE
    b[2][2] = WHITE
    g = GoGame.from_position(b, to_move=BLACK)
    assert g.place_stone(1, 2) is True
    assert g.get_board()[1][1] == EMPTY
    assert g.captures[BLACK] == 1
    assert g.to_move == WHITE
    assert g.is_legal(1, 1) is False
    assert g.place_stone(1, 1) is False
    assert g.get_board()[1][2] == BLACK


def test_ko_lifts_after_interim_move():
    b = blank_board()
    b[0][1] = BLACK
    b[1][0] = BLACK
    b[2][1] = BLACK
    b[0][2] = WHITE
    b[1][1] = WHITE
    b[1][3] = WHITE
    b[2][2] = WHITE
    g = GoGame.from_position(b, to_move=BLACK)
    assert g.place_stone(1, 2) is True
    assert g.place_stone(8, 8) is True  # White plays elsewhere
    assert g.place_stone(8, 7) is True  # Black plays elsewhere
    assert g.is_legal(1, 1) is True
    assert g.place_stone(1, 1) is True
    assert g.get_board()[1][2] == EMPTY
    assert g.captures[WHITE] == 1


def test_is_legal_does_not_mutate():
    b = blank_board()
    b[0][0] = BLACK
    b[1][0] = WHITE
    g = GoGame.from_position(b, to_move=WHITE)
    snapshot_board = copy.deepcopy(g.get_board())
    snapshot_captures = dict(g.captures)
    snapshot_to_move = g.to_move
    for r in range(SIZE):
        for c in range(SIZE):
            g.is_legal(r, c)
    assert g.get_board() == snapshot_board
    assert g.captures == snapshot_captures
    assert g.to_move == snapshot_to_move


def test_out_of_bounds_rejected():
    g = GoGame()
    assert g.is_legal(-1, 0) is False
    assert g.is_legal(0, -1) is False
    assert g.is_legal(SIZE, 0) is False
    assert g.is_legal(0, SIZE) is False
    assert g.place_stone(-1, 0) is False
    assert g.place_stone(SIZE, SIZE) is False
    assert g.to_move == BLACK  # no turn consumed on illegal


def test_occupied_rejected():
    g = GoGame()
    assert g.place_stone(4, 4) is True
    assert g.is_legal(4, 4) is False
    assert g.place_stone(4, 4) is False
    assert g.to_move == WHITE


def test_two_passes_end_game():
    g = GoGame()
    g.place_stone(4, 4)  # Black
    assert g.to_move == WHITE
    g.pass_turn()  # White passes
    assert g.finished is False
    assert g.to_move == BLACK
    g.pass_turn()  # Black passes
    assert g.finished is True
    # Board has 1 black stone and 80 empty cells touching only black.
    # Score: Black = 1 stone + 80 territory = 81.
    # White = 0 stones + 0 territory + 2.5 komi = 2.5.
    result = g.score()
    assert result["black"] == 81
    assert result["white"] == 2.5
    assert result["winner"] == BLACK


def test_concede_ends_game_immediately():
    g = GoGame()
    g.place_stone(4, 4)  # Black
    assert g.to_move == WHITE
    g.concede()  # White concedes
    assert g.finished is True
    assert g.loser == WHITE
    result = g.score()
    assert result["winner"] == BLACK


def test_history_records_moves_passes_and_concede():
    g = GoGame()
    g.place_stone(4, 4)
    g.place_stone(3, 3)
    g.pass_turn()
    g.place_stone(5, 5)
    assert g.history == [(4, 4), (3, 3), "pass", (5, 5)]
    # clone_fast preserves history.
    g2 = g.clone_fast()
    assert g2.history == g.history
    # concede is recorded too.
    g3 = GoGame()
    g3.place_stone(0, 0)
    g3.concede()
    assert g3.history == [(0, 0), "concede"]


def test_score_empty_board():
    g = GoGame()
    r = g.score()
    assert r["black"] == 0
    assert r["white"] == pytest.approx(KOMI)
    assert r["black_territory"] == 0
    assert r["white_territory"] == 0
    assert r["neutral_points"] == SIZE * SIZE
    assert r["winner"] == WHITE


def test_score_clean_territory_split():
    # Black wall at col 3, White wall at col 5. Col 4 is dame.
    b = blank_board()
    for r in range(SIZE):
        b[r][3] = BLACK
        b[r][5] = WHITE
    g = GoGame.from_position(b)
    r = g.score()
    # Cols 0-2 (27 cells) are Black territory; cols 6-8 (27 cells) are White.
    # Col 4 (9 cells) touches both -> dame, not counted.
    assert r["black_stones"] == SIZE
    assert r["white_stones"] == SIZE
    assert r["black_territory"] == 27
    assert r["white_territory"] == 27
    assert r["neutral_points"] == 9
    assert r["black"] == SIZE + 27
    assert r["white"] == pytest.approx(SIZE + 27 + KOMI)
    assert r["winner"] == WHITE


def test_score_dame_not_counted():
    # One Black and one White stone far apart on an otherwise empty board.
    # The entire empty region touches both colors -> dame, not counted.
    b = blank_board()
    b[0][0] = BLACK
    b[SIZE - 1][SIZE - 1] = WHITE
    g = GoGame.from_position(b)
    r = g.score()
    assert r["black_territory"] == 0
    assert r["white_territory"] == 0
    assert r["neutral_points"] == SIZE * SIZE - 2
    assert r["black_stones"] == 1
    assert r["white_stones"] == 1
    assert r["black"] == 1
    assert r["white"] == pytest.approx(1 + KOMI)
