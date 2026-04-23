# Go Terms Dictionary

A plain-English glossary of every Go term used in this project's code, tests, and docs. Read this first if you're new to Go — the rest of the docs assume these terms.

Legend for the diagrams below:

- `B` = Black stone
- `W` = White stone
- `.` = empty intersection
- Diagrams are small slices of a 9×9 board; rows/cols grow down/right.

---

## Board & pieces

### Intersection (point)
A place where two lines cross on the board. Stones sit **on** intersections, not inside squares. A 9×9 Go board has `9 * 9 = 81` intersections.

### Stone
A single game piece. Black or White. Once placed, a stone does not move — it can only be captured and removed.

### Empty
An intersection with no stone. Represented as `0` / `EMPTY` in code.

### Color / player
Either Black or White. Black always plays first.

---

## Basic mechanics

### Liberty
An **empty** intersection directly adjacent (up, down, left, or right — **no diagonals**) to a stone. Liberties are the "breathing room" a stone or group needs to stay alive.

Example — the single Black stone at `(1,1)` has **4 liberties** (all 4 neighbors are empty):

```
. . .
. B .
. . .
```

Same stone pushed into the corner `(0,0)` has only **2 liberties** — the other two "neighbors" are off the board:

```
B . .
. . .
```

### Adjacent / neighbor
The 4 orthogonal points (up, down, left, right). Diagonals do **not** count as adjacent in Go. In code, `_NEIGHBOR_OFFSETS = ((-1,0),(1,0),(0,-1),(0,1))`.

### Group (chain)
A connected cluster of same-colored stones linked through adjacency. A group shares its liberties — any empty neighbor of **any** stone in the group is a liberty for the **whole** group.

Example — the two Black stones form one group because they're adjacent. Together they have **3 liberties** (the three empty neighbors around the pair; the internal shared edge is not a liberty because it isn't empty on the outside):

```
. . . .
. B B .
. . . .
```

### Capture
When a group's liberty count drops to zero, every stone in that group is removed from the board. The player who placed the stone that caused the capture gains points equal to the number of stones removed.

Example — it is White's turn. White plays at `(1,2)`:

```
Before:            After White plays (1,2):
. B .               . B .
B W .               B . W
. B .               . B .
```

The White stone at `(1,1)` had only one liberty left (at `(1,2)`). Playing there removes it.

---

## The three rules that make Go interesting

### Suicide
Playing a stone into a position where your own group would have **zero liberties** — and the move doesn't capture anything — is illegal.

Example — Black may **not** play at `(0,0)` (all neighbors are White, nothing gets captured):

```
. W . .
W . . .
. . . .
```

### Capture-first rule (why "self-atari that captures" is legal)
When a stone is placed, the engine resolves in this order:

1. Remove any **opponent** groups that now have zero liberties.
2. **Then** check whether the played stone's own group has any liberties.

This ordering matters: a move that would appear to be suicide can be legal if it captures an opponent group first, which frees up liberties for your stone.

Example — it is Black's turn. Black plays at `(4,4)`. All 4 neighbors are White, but the White stone at `(3,4)` has exactly one liberty (`(4,4)`). The capture-first rule removes `(3,4)` **before** checking Black's liberties, and Black ends up with 1 liberty at `(3,4)`:

```
Before:                    After Black plays (4,4):
. . B . .                  . . B . .
. B W B .                  . B . B .
. W B W .                  . W B W .
. . W . .                  . . W . .
```

### Ko (the repetition rule)
Literally "eternity" in Japanese. Without a repetition rule, two players could capture and recapture the same single stone forever. The **simple ko** rule used in this engine: a move is illegal if it would recreate the **board position from immediately before the opponent's last move**.

Example — Black has just captured a White stone at `(1,1)`. White is not allowed to immediately recapture at `(1,2)` on the very next move, because that would put the board back where it was one move ago.

```
Before Black's move:       After Black captures:      White CANNOT play (1,2) next:
. B W .                    . B W .                    (would revert to the 
B W . W                    B . B W                     "before" position)
. B W .                    . B W .
```

If White plays somewhere else first, the ban lifts and the point becomes legal again.

---

## Special situations

### Snapback
A move that looks like self-atari (one-liberty play) but is actually good because it immediately captures a larger group. The capture-first rule is what makes snapback legal.

### Atari
A group with exactly **one** liberty left. One more move on that liberty captures it. (Informational term — not used in the code directly, but useful when describing game states.)

### Dame (neutral point)
An empty intersection (or connected empty region) that borders **both** colors, or neither. It counts for nobody at scoring time.

Example — col 4 is dame because the empty region between the two walls touches both Black and White:

```
. . . B . W . . .
. . . B . W . . .
. . . B . W . . .
...
```

### Territory
An empty region bordered by **only one color**. That whole region counts as points for that color.

In the diagram above, cols 0–2 are Black's territory (they touch only Black), and cols 6–8 are White's territory.

### Eye
A single empty point surrounded by one color's stones on all sides — essentially a permanent liberty. A group with two separate eyes can never be captured because the opponent can't fill both at once without suiciding. The AI (Day 2) uses a "don't fill your own eyes" heuristic during rollouts.

Example — Black has one eye at `(1,1)`:

```
B B B
B . B
B B B
```

---

## Ending the game & scoring

### Pass
A player can pass instead of placing a stone. **In this project, passing means conceding** — the passer loses immediately. (Standard Go uses two consecutive passes to end the game; the assignment spec overrides this.)

### Komi
Compensation points given to White to offset Black's first-move advantage. This project uses `KOMI = 2.5`. The half-point guarantees there are no ties.

### Chinese area scoring
The scoring method used in this project:

- **Black's score** = Black stones on the board + empty points that are Black's territory
- **White's score** = White stones on the board + empty points that are White's territory + komi
- Dame points count for nobody.
- Higher total wins. If someone passed/conceded, they lose regardless of area.

### Winner
Whoever has the higher score after `score()` runs. If a player conceded via `pass_turn()`, the opponent wins automatically.

---

## Implementation terms (specific to this codebase)

### Flat board
Instead of a 2D list, the engine stores the 9×9 board as a 1D list of 81 ints. Position `(row, col)` lives at index `row * 9 + col`. Flat storage lets us cheaply turn the whole board into a tuple for ko comparison.

### Snapshot (`pre_move`)
A tuple copy of the board taken **before** a move is tentatively applied. If the move turns out to be illegal (suicide or ko), `self.board = list(pre_move)` undoes everything — both the placed stone and any captures we did along the way — in one line.

### `prev_position`
The snapshot from one move ago, kept around to implement the simple-ko check. A new move is illegal if playing it would recreate `prev_position`.

### BFS (breadth-first search)
The search strategy used by `_group` and `_territory` to explore connected regions. Start from one point, expand outward one step at a time using a queue, marking visited cells so we don't revisit them.

### `finished` / `loser`
Two flags set when somebody passes. `finished = True` blocks any further moves; `loser` records who passed so `score()` can award the win to the opponent.

### `to_move`
Whose turn it is right now. Starts as Black, flips after every legal move.

### `captures`
A dict `{BLACK: int, WHITE: int}` counting how many stones each color has captured across the game so far. The GUI displays these.

### `from_position` (test-only)
A classmethod on `GoGame` that builds a game at an arbitrary board state without replaying moves. Used only by tests to set up tricky positions like ko and snapback directly. Production code paths (MCTS, GUI) must never use it.

---

## Quick cheat-sheet

| Term | One-line meaning |
|---|---|
| Liberty | empty neighbor of a stone/group |
| Group | connected same-colored stones |
| Capture | remove a group with 0 liberties |
| Suicide | playing into 0 liberties without capturing (illegal) |
| Ko | can't recreate the position from one move ago |
| Snapback | legal "self-atari" that captures first |
| Atari | group with exactly 1 liberty |
| Dame | neutral empty region, counts for nobody |
| Territory | empty region bordered by one color only |
| Eye | empty point fully surrounded by one color |
| Pass | **in this project, = concede** |
| Komi | White's 2.5-point handicap |
