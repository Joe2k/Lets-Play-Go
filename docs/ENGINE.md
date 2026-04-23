# Rules Engine Walkthrough

Plain-English tour of [engine/go_engine.py](../engine/go_engine.py). One section per function, each with what it does, why it exists, and an input/output table.

---

## Module-level constants

| Name | Value | Meaning |
|---|---|---|
| `SIZE` | `9` | Board is 9×9 |
| `EMPTY` | `0` | Empty intersection |
| `BLACK` | `1` | Black stone |
| `WHITE` | `2` | White stone |
| `KOMI` | `2.5` | Bonus points White gets at scoring |
| `_NEIGHBOR_OFFSETS` | `((-1,0),(1,0),(0,-1),(0,1))` | The 4 orthogonal neighbors (up, down, left, right). Go has no diagonals. |

**Board layout:** stored as a flat list of 81 ints. Cell `(row, col)` lives at index `row * 9 + col`. Flat is cheaper to hash as a tuple, which matters for the ko check.

---

## `_other(color)`

**What it does:** flips a color. Black → White, White → Black.

| Input | Output |
|---|---|
| `_other(BLACK)` | `WHITE` |
| `_other(WHITE)` | `BLACK` |

**Why:** we use it to get "the opponent" when resolving captures and when deciding the winner after a concession.

---

## `GoGame.__init__()`

**What it does:** creates a new game by calling `new_game()`.

| Input | Effect |
|---|---|
| `GoGame()` | fresh empty board, Black to move |

---

## `GoGame.new_game()`

**What it does:** resets every piece of game state back to the start.

Fields it sets:

| Field | Reset to | Meaning |
|---|---|---|
| `board` | `[0] * 81` | all empty |
| `to_move` | `BLACK` | Black plays first |
| `prev_position` | `None` | no previous position yet (used for ko) |
| `captures` | `{BLACK: 0, WHITE: 0}` | nobody has captured anything |
| `last_move` | `None` | no move played yet (used by the GUI to highlight) |
| `finished` | `False` | game is active |
| `loser` | `None` | nobody has lost yet |

---

## `GoGame.place_stone(row, col)` — the heart of the engine

**What it does:** tries to play a stone at `(row, col)` for the current player. If the move is legal, it applies captures, updates whose turn it is, and returns `True`. If illegal, it leaves the game untouched and returns `False`.

**Why rule order matters:** Go says "capture opponents first, THEN look at your own stone's liberties." This is what makes the snapback / legal-self-atari cases work (see test 8). If you checked your own suicide first, a move that captures an opponent group and saves itself would be wrongly rejected.

### Step-by-step

| # | Step | What it checks / does |
|---|---|---|
| 1 | game over? | if `finished`, return `False` |
| 2 | in bounds? | reject off-board coordinates |
| 3 | cell empty? | reject plays onto an occupied point |
| 4 | snapshot | save a tuple of the board *before* the move (used to undo if illegal, and used as the next ko comparand) |
| 5 | place stone | put the mover's color at `(row, col)` |
| 6 | capture opponents | for each of the 4 neighbors that holds an opponent stone, find its group; if the group has zero liberties, remove it and add to capture count |
| 7 | own suicide? | find the mover's group and its liberties; if zero, restore the snapshot and return `False` |
| 8 | ko? | if the board now equals `prev_position`, restore the snapshot and return `False` (simple-ko rule) |
| 9 | commit | store the pre-move snapshot as the new `prev_position`, add captures, remember `last_move`, flip `to_move`, return `True` |

### Returns

| Situation | Return |
|---|---|
| Move applied | `True` |
| Any illegal condition (off-board, occupied, suicide, ko, game over) | `False` |

### Key idea: the snapshot

The `pre_move` tuple is the engine's "undo button". If we discover mid-function that the move is illegal (suicide or ko), we just do `self.board = list(pre_move)` and **everything** — the placed stone AND any opponent captures we already did — is rolled back in one line.

---

## `GoGame.is_legal(row, col)`

**What it does:** tells you whether `place_stone(row, col)` *would* succeed, without actually playing the move.

**How:** deep-copies the game and calls `place_stone` on the copy. Whatever that returns is the answer. This guarantees `is_legal` and `place_stone` can never drift apart — there is only one implementation of the rules.

| Input | Output |
|---|---|
| Legal move | `True` |
| Off-board / occupied / suicide / ko / game over | `False` |

Fast-path checks (game finished, out of bounds, cell not empty) happen before the deepcopy as a minor optimization.

---

## `GoGame.get_board()`

**What it does:** returns the board as a fresh 9×9 nested list (a copy, not a reference).

| Input | Output |
|---|---|
| `get_board()` | `list[list[int]]` of shape `9x9` with values in `{0, 1, 2}` |

**Why a copy:** callers (the GUI, tests, the AI) can read or mutate what they get back without accidentally corrupting the engine's internal flat list.

---

## `GoGame.pass_turn()`

**What it does:** the current player concedes. The game ends immediately and they lose.

| Field after call | Value |
|---|---|
| `finished` | `True` |
| `loser` | whoever was `to_move` |
| `last_move` | `"pass"` |

Calling `pass_turn()` on an already-finished game is a no-op.

**Why "passer loses" instead of normal Go:** that's the spec for this assignment. Real Go uses two consecutive passes and then scores; here, a single pass = concession.

---

## `GoGame.end_game()`

**What it does:** alias for `pass_turn()`. Exists because the spec lists `end_game` as part of the required public API.

---

## `GoGame.score()`

**What it does:** computes Chinese area scoring and returns a result dictionary.

### Scoring rules used

- **Black points** = Black stones on the board + empty regions that border only Black
- **White points** = White stones on the board + empty regions that border only White + `KOMI` (2.5)
- **Dame** (empty regions touching both colors, or touching no stones at all) count for nobody
- **Winner**: if someone conceded via `pass_turn`, the opponent wins automatically. Otherwise whoever has more points wins. (Ties go to White because of komi, and the current implementation breaks ties toward White as well.)

### Return dict

| Key | Meaning |
|---|---|
| `black` | Black's total (stones + territory) |
| `white` | White's total (stones + territory + komi) |
| `black_stones` | Black stones on the board |
| `white_stones` | White stones on the board |
| `black_territory` | empty points owned by Black |
| `white_territory` | empty points owned by White |
| `komi` | `2.5` (included so the GUI can show it) |
| `winner` | `BLACK` or `WHITE` |

---

## `GoGame.from_position(...)` — test-only constructor

**What it does:** builds a `GoGame` instance at an arbitrary board position without having to play moves to get there.

**Why:** setting up a ko or a snapback by hand would otherwise require playing 10+ legal moves in the right order. This classmethod skips that.

| Parameter | Meaning |
|---|---|
| `board` | 9×9 nested list of `0/1/2` |
| `to_move` | `BLACK` or `WHITE` (defaults to Black) |
| `prev_position` | optional 9×9 position, used to set up ko-specific tests |
| `captures` | optional `{BLACK: int, WHITE: int}` starting capture counts |

Validates shape (must be 9×9), cell values (must be in `{0,1,2}`), and `to_move` (must be `BLACK` or `WHITE`). Raises `ValueError` on bad input.

**Not for production.** MCTS and the GUI must always use `GoGame()` + `place_stone(...)` so the game stays reachable from the starting position. This is why the docstring explicitly says "TEST ONLY".

---

## `GoGame._group(row, col)` — private helper

**What it does:** finds the connected group of same-colored stones that contains `(row, col)`, plus its liberties.

Uses BFS (breadth-first search) starting from `(row, col)`. For each stone in the group:
- a same-colored neighbor gets added to the group
- an empty neighbor gets added to the liberty set
- an opponent or off-board neighbor is ignored

| Input | Output |
|---|---|
| `(r, c)` on an empty cell | `(empty set, empty set)` |
| `(r, c)` on a stone | `(set of (r,c) in the group, set of (r,c) empty neighbors)` |

**Why we need liberties as a set, not a count:** two stones in the same group can share a liberty, but it should only be counted once. Using a set deduplicates automatically.

---

## `GoGame._remove(stones)` — private helper

**What it does:** empties every `(r, c)` in the given set. Used to remove captured groups.

| Input | Effect |
|---|---|
| `{(r1,c1), (r2,c2), ...}` | those cells become `EMPTY` |

---

## `GoGame._territory()` — private helper

**What it does:** figures out which empty points belong to Black, which belong to White, and which are dame (neutral).

### Algorithm

For each empty point not yet visited:
1. BFS-flood outward, collecting all connected empty cells into one *region*.
2. Every time the flood touches a stone, record that stone's color in a `borders` set.
3. After the region is fully mapped:
   - if `borders == {BLACK}` → the whole region is Black's territory
   - if `borders == {WHITE}` → the whole region is White's territory
   - anything else (`{BLACK, WHITE}` or empty) → dame, counts for nobody

| Returns | Meaning |
|---|---|
| `(black_t, white_t)` | territory counts |

**Why BFS-by-region instead of scanning cells individually:** one empty cell's owner depends on the whole connected region it's in, not just its immediate neighbors. A single empty point next to a Black stone might still be dame if, ten cells away through empty space, the same region also touches a White stone.

---

## Summary: why this design is small on purpose

- **One source of truth.** Every legality question routes through `place_stone`. `is_legal` is just a deepcopy probe. No duplicated rule code means rules can't drift.
- **Flat board + snapshot undo.** Storing the board as a flat list lets us `tuple(self.board)` cheaply for the ko check, and `self.board = list(pre_move)` is a one-line undo.
- **Capture-first rule ordering.** Captures resolve before the mover's own-liberty check, which handles snapback correctly without any special-case code.
- **Chinese area scoring fits in ~25 lines.** Stones + single-color-bordered empty regions + komi. No dead-stone marking needed because the spec uses pass-to-concede.
