# Rules Engine Test Guide

This document walks through every test in [tests/test_engine.py](../tests/test_engine.py) in plain language. Each test pins down one specific Go rule so that a failure points directly at what broke.

Board conventions used throughout:

- `9x9` board, `B` = Black, `W` = White, `.` = empty
- Coordinates are `(row, col)`, `0`-indexed from the top-left
- `KOMI = 2.5` (awarded to White under Chinese area scoring)
- Passing follows standard Go: two consecutive passes end the game and territory is scored. `concede()` exists separately for immediate resignation.

---

## 1. `test_new_game_empty`

**Rule:** A new game starts with an empty 9×9 board, Black to move, zero captures, not finished.

| Input | Expected |
|---|---|
| `GoGame()` | board is 9×9 all empty, `to_move == BLACK`, `captures == {B:0, W:0}`, `finished is False` |

---

## 2. `test_simple_placement_and_turn_swap`

**Rule:** A legal move places the mover's stone and flips the turn.

| Step | Input | Expected |
|---|---|---|
| 1 | `place_stone(4, 4)` (Black) | returns `True`, `board[4][4] == BLACK`, `to_move == WHITE` |
| 2 | `place_stone(3, 3)` (White) | returns `True`, `board[3][3] == WHITE`, `to_move == BLACK` |

---

## 3. `test_single_stone_capture`

**Rule:** A stone with no liberties is removed from the board; the capturer's capture count goes up.

Setup (Black stone at corner with one White neighbor, White to play the killing move):

```
B . . ...
W . . ...
. . . ...
```

| Input | Expected |
|---|---|
| White `place_stone(0, 1)` | `board[0][0]` becomes `EMPTY` (Black captured), `board[0][1] == WHITE`, `captures[WHITE] == 1` |

---

## 4. `test_edge_capture_two_stone_group`

**Rule:** Captures work on connected groups, not just single stones. Edge groups have fewer liberties.

Setup (two Black stones along the top edge, White surrounds from below and the right):

```
B B W ...
W . . ...
. . . ...
```

| Input | Expected |
|---|---|
| White `place_stone(1, 1)` | both Black stones at `(0,0)` and `(0,1)` removed, `captures[WHITE] == 2` |

---

## 5. `test_multi_stone_capture_center`

**Rule:** Captures work in the middle of the board too, on a larger group.

Setup (a 2-stone Black group at the center, fully ringed by White except one spot):

```
. . . . . . . . .
. . . . . . . . .
. . . . . . . . .
. . . . W . . . .
. . . W B W . . .
. . . W B W . . .
. . . . . . . . .
...
```

| Input | Expected |
|---|---|
| White `place_stone(6, 4)` | both Black stones at `(4,4)` and `(5,4)` removed, `captures[WHITE] == 2` |

---

## 6. `test_suicide_rejected_single_stone`

**Rule:** Playing a stone that would have zero liberties AND captures nothing is illegal (suicide).

Setup (Black tries to play into the corner, all neighbors are White, no White group dies):

```
. W W ...
W . . ...
W . . ...
```

| Input | Expected |
|---|---|
| Black `is_legal(0, 0)` | `False` |
| Black `place_stone(0, 0)` | `False`, board unchanged |

---

## 7. `test_suicide_rejected_multi_stone`

**Rule:** Suicide is still illegal when the would-be-dead shape is a multi-stone group.

Setup (a Black stone already at `(0,1)`; Black at `(0,0)` would join it into a 2-stone group with no liberties):

```
. B W ...
W W . ...
. . . ...
```

| Input | Expected |
|---|---|
| Black `is_legal(0, 0)` | `False` |
| Black `place_stone(0, 0)` | `False`, existing `B` at `(0,1)` still there |

---

## 8. `test_legal_self_atari_that_captures`

**Rule (the snapback rule / capture-first):** A move that would otherwise be suicide is **legal** if it captures at least one opponent group first. Captures are resolved before the mover's own-liberty check.

Setup (Black plays `(4,4)` — all 4 neighbors are White, but the White stone at `(3,4)` has only `(4,4)` as its liberty):

```
. . . . . . . . .
. . . . . . . . .
. . . . B . . . .
. . . B W B . . .
. . . W . W . . .
. . . . W . . . .
. . . . . . . . .
...
```

| Input | Expected |
|---|---|
| Black `is_legal(4, 4)` | `True` |
| Black `place_stone(4, 4)` | `True`, `board[3][4]` becomes `EMPTY` (W captured), `board[4][4] == BLACK`, `captures[BLACK] == 1` |

---

## 9. `test_ko_basic`

**Rule (simple ko):** A move is illegal if it recreates the board position from immediately before the opponent's previous move. Prevents an infinite single-stone recapture loop.

Setup (classic 1-stone ko shape; Black captures first):

```
. B W . . . . . .
B W . W . . . . .
. B W . . . . . .
...
```

| Step | Input | Expected |
|---|---|---|
| 1 | Black `place_stone(1, 2)` | `True`, captures W at `(1,1)`, `captures[BLACK] == 1`, `to_move == WHITE` |
| 2 | White `is_legal(1, 1)` | `False` (would recreate pre-step-1 position) |
| 3 | White `place_stone(1, 1)` | `False`, `board[1][2]` still `BLACK` |

---

## 10. `test_ko_lifts_after_interim_move`

**Rule:** The ko ban only applies to the *immediate* recapture. After any other move is played, the ko point becomes legal again.

| Step | Input | Expected |
|---|---|---|
| 1 | Black `place_stone(1, 2)` | `True` (captures ko stone) |
| 2 | White `place_stone(8, 8)` | `True` (elsewhere) |
| 3 | Black `place_stone(8, 7)` | `True` (elsewhere) |
| 4 | White `is_legal(1, 1)` | `True` (ko ban lifted) |
| 5 | White `place_stone(1, 1)` | `True`, captures Black at `(1,2)`, `captures[WHITE] == 1` |

---

## 11. `test_is_legal_does_not_mutate`

**Rule (invariant):** `is_legal(r, c)` must be a pure query — it cannot modify the board, capture counts, or whose turn it is.

| Input | Expected |
|---|---|
| Call `is_legal(r, c)` for every cell on the board | board, captures, and `to_move` all match the snapshot taken before the loop |

---

## 12. `test_out_of_bounds_rejected`

**Rule:** Coordinates outside `[0, 9)` are rejected by both `is_legal` and `place_stone`. An illegal attempt does **not** consume a turn.

| Input | Expected |
|---|---|
| `is_legal(-1, 0)`, `is_legal(0, -1)`, `is_legal(9, 0)`, `is_legal(0, 9)` | all `False` |
| `place_stone(-1, 0)`, `place_stone(9, 9)` | both `False` |
| after all attempts | `to_move == BLACK` (turn not consumed) |

---

## 13. `test_occupied_rejected`

**Rule:** You cannot play on a non-empty intersection.

| Step | Input | Expected |
|---|---|---|
| 1 | Black `place_stone(4, 4)` | `True`, now `to_move == WHITE` |
| 2 | White `is_legal(4, 4)` | `False` |
| 3 | White `place_stone(4, 4)` | `False`, `to_move` still `WHITE` |

---

## 14a. `test_two_passes_end_game`

**Rule:** Two consecutive `pass_turn()` calls end the game. The final position is scored normally (territory + komi).

| Step | Input | Expected |
|---|---|---|
| 1 | Black `place_stone(4, 4)` | `True`, `to_move == WHITE` |
| 2 | White `pass_turn()` | `finished is False`, `to_move == BLACK` |
| 3 | Black `pass_turn()` | `finished is True` |
| 4 | `score()` | `black == 81` (1 stone + 80 territory), `white == 2.5` (komi only), `winner == BLACK` |

## 14b. `test_concede_ends_game_immediately`

**Rule:** `concede()` ends the game immediately and the conceder loses, regardless of board position.

| Step | Input | Expected |
|---|---|---|
| 1 | Black `place_stone(4, 4)` | `True`, `to_move == WHITE` |
| 2 | White `concede()` | `finished is True`, `loser == WHITE` |
| 3 | `score()` | `winner == BLACK` |

## 14c. `test_history_records_moves_passes_and_concede`

**Rule:** `GoGame.history` appends every action in order — stone placements as `(row, col)`, passes as `"pass"`, concessions as `"concede"`. External-engine adapters (`GnuGoAgent`) replay this list to sync state.

---

## 15. `test_score_empty_board`

**Rule:** On an empty board, nobody owns any territory (the empty region borders no stones), so White wins on komi alone.

| Input | Expected |
|---|---|
| `GoGame().score()` | `black == 0`, `white == 2.5` (just komi), `black_territory == 0`, `white_territory == 0`, `winner == WHITE` |

---

## 16. `test_score_clean_territory_split`

**Rule (Chinese area scoring):** An empty region that borders only one color counts as that player's territory. Regions touching both colors are *dame* and count for nobody.

Setup (Black wall on col 3, White wall on col 5; col 4 is a dame strip):

```
. . . B . W . . .
. . . B . W . . .
. . . B . W . . .
... (all 9 rows the same pattern)
```

| Input | Expected |
|---|---|
| `score()` | `black_stones == 9`, `white_stones == 9`, `black_territory == 27` (cols 0-2), `white_territory == 27` (cols 6-8), col 4 (9 cells) is dame and uncounted, `black == 36`, `white == 38.5` (36 + komi), `winner == WHITE` |

---

## 17. `test_score_dame_not_counted`

**Rule:** When the entire empty region touches both colors, it is all dame — neither side gets any territory points.

Setup (one Black stone at `(0,0)`, one White stone at `(8,8)`, everything else empty):

```
B . . . . . . . .
. . . . . . . . .
...
. . . . . . . . W
```

| Input | Expected |
|---|---|
| `score()` | `black_territory == 0`, `white_territory == 0`, `black_stones == 1`, `white_stones == 1`, `black == 1`, `white == 3.5` (1 + komi) |

---

## Running the tests

From the repo root with the project's `.venv` activated:

```bash
pytest -v tests/
```

All 17 tests should pass. If one fails, the test name tells you which rule regressed.
