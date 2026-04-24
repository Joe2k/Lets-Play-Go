# MCTS AI Walkthrough

Plain-English tour of [ai/mcts.py](../ai/mcts.py) and [ai/agent.py](../ai/agent.py). One section per function/class, each with what it does, why it exists, and an input/output table.

If you don't yet know what a "liberty" or an "eye" is, read [GLOSSARY.md](GLOSSARY.md) first — this doc assumes those terms.

---

## What MCTS is, in one paragraph

**Monte Carlo Tree Search** is "play a lot of random games from here, see which first move wins most often, and pick that one." The clever part is that it doesn't just sample uniformly — it builds a tree of explored positions and uses a formula (UCB1) to spend more samples on moves that look promising while still occasionally exploring others. Each MCTS iteration does four things: **select** a path through the tree, **expand** one new node, **simulate** a random game from that node to a scored end, and **backpropagate** the result up the path. Over thousands of iterations, the win rates settle and the most-visited move at the root is the answer.

---

## Module constants (in `ai/mcts.py`)

| Name | Value | Meaning |
|---|---|---|
| `_NEIGHBORS` | `((-1,0),(1,0),(0,-1),(0,1))` | The 4 orthogonal directions. Same as the engine's. |
| `_DEFAULT_C` | `sqrt(2) ≈ 1.414` | UCB1 exploration constant. Textbook default. |
| `_ROLLOUT_MOVE_CAP` | `200` | Hard limit on moves per random rollout (safety net against pathological loops). |

---

## `_is_own_eye(board_flat, r, c, color)`

**What it does:** returns `True` if `(r, c)` is empty AND every in-bounds 4-neighbor is the same `color`.

| Input | Output |
|---|---|
| Point `(r,c)` is empty and all 4 neighbors are `color` | `True` |
| Point is occupied OR any neighbor is a different color OR neighbor is empty | `False` |

**Why:** the "don't fill your own eyes" heuristic. Without it, random rollouts kill their own alive groups by filling eyes, and the win-rate signal becomes useless. This is the **single most important** detail for making random-rollout MCTS work in Go.

**Why 4 neighbors only (no diagonals):** the textbook eye rule also looks at diagonal corners — a true eye usually wants ≥3 same-colored diagonals. We use the simpler 4-neighbor version because (a) it's easy to defend in the oral, (b) it's good enough on a 9×9 board, and (c) the few false positives it produces just mean we sometimes pass when we could play a (probably useless) move. The downside (occasionally passing when winning) is a known trade-off.

---

## `candidate_moves(game)`

**What it does:** returns a list of `(row, col)` coordinates that are worth considering for the player to move — empty points that aren't own-eyes.

| Input | Output |
|---|---|
| `GoGame` instance | `list[tuple[int, int]]` of candidate moves |
| Empty board, Black to move | ~80 candidates (skips none — no eyes yet) |
| Position where every empty is Black's own eye, Black to move | `[]` |

**What it deliberately does NOT check:** legality. Suicide and ko moves slip through. Why? `is_legal` deepcopies the game (very slow), and we call `candidate_moves` thousands of times during search. The downstream code (`_expand` and `_simulate`) catches illegal moves cheaply by checking `place_stone`'s return value. This shaved roughly 10× off our hot-path cost.

---

## `class Node`

**What it represents:** one position in the MCTS tree. Each node corresponds to a board state reached by a specific sequence of moves from the root.

### Fields

| Field | Meaning |
|---|---|
| `parent` | the node above this one (None at root) |
| `move` | the `(r, c)` that was played to reach this node from `parent` (None at root) |
| `to_move` | which color is to move at this state |
| `children` | dict mapping `move -> Node` for replies we've explored |
| `untried_moves` | candidate moves we haven't expanded yet |
| `visits` | how many times this node has been on a simulation path |
| `wins` | how many of those simulations were won by the player who chose this node (i.e., `parent.to_move`) |
| `is_terminal` | True if there are no candidate moves (or all turned out illegal during expansion) |

### `is_fully_expanded()`

| Returns | Meaning |
|---|---|
| `True` | every untried move has been turned into a child node already |
| `False` | there's at least one untried move we could expand |

### `best_child_uct(c)`

**What it does:** picks the child that maximizes UCB1 — the "which next move should we explore?" choice during selection.

```
UCB1 = (wins / visits) + c * sqrt(ln(parent.visits) / visits)
        ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        exploitation       exploration (bigger when this child is under-visited)
```

| Input | Output |
|---|---|
| Exploration constant `c` | child Node with highest UCB1 score |

**Why this formula:** the first term picks moves with high win rates (exploit what we know). The second term picks moves with few visits relative to the parent (explore what we don't). Together they balance the two without a hard switch. `c = sqrt(2)` is the value the original UCB1 paper proves regret-optimal.

---

## The four MCTS phases (private helpers)

### `_select(root, game, c)`

**What it does:** walks down the tree from the root using UCB1, mutating `game` to reflect each move played. Stops when it hits a node that's terminal, has untried moves, or has no children at all.

| Input | Output |
|---|---|
| `(root_node, deep-copied game, exploration constant c)` | `(leaf_node, game now reflecting the path's moves)` |

The leaf returned is where expansion will happen next.

### `_expand(node, game, rng)`

**What it does:** if `node` has untried moves, pick one randomly, try to play it, and create a child node. If `place_stone` rejects the move (suicide or ko slipped through the cheap filter), drop it from the untried list and try another. If we exhaust them all, mark the node terminal.

| Input | Output |
|---|---|
| `(node, game, rng)` | `(child_or_self, updated game)` |
| Node has at least one legal untried move | new child created and returned |
| All untried moves are illegal | node marked terminal, returned unchanged |

### `_simulate(game, rng)`

**What it does:** plays a random game from the current state to a scored end. On each turn, gets candidate moves, shuffles them, and plays the first one `place_stone` accepts. Stops when (a) no candidate moves remain, or (b) the move cap of 200 is hit.

| Input | Output |
|---|---|
| `(game, rng)` | `BLACK` or `WHITE` (the rollout's winner per `score()`) |

**Important:** the rollout never calls `pass_turn()`. Passing in this project means losing, so calling it during a rollout would destroy the win-rate signal. Instead, when no moves are available we just stop and call `score()` directly. The "passer loses" override only kicks in if `loser` is set, which it never is during a rollout.

### `_backprop(node, winner)`

**What it does:** walks from the leaf back up to the root, incrementing `visits` on every node and incrementing `wins` on nodes whose mover won.

| Input | Effect on each node up the path |
|---|---|
| `(leaf_node, winner)` | `visits += 1`; `wins += 1` if `parent.to_move == winner` |

**Why "if parent.to_move == winner":** each node represents a state reached by a move chosen by its parent's player. So the win count at a node means "how often did the player who chose this node win from here." That's the value we want when deciding which child to pick at the parent.

---

## `search(root_game, root, iterations, c, rng)`

**What it does:** runs `iterations` MCTS iterations from `root`, mutating `root` in place. Each iteration deep-copies `root_game`, then runs the four phases.

| Input | Effect |
|---|---|
| `(starting position, root node, N, c, rng)` | `root` and its descendants accumulate visits/wins from N rollouts |

**Why deepcopy per iteration:** `_select` and the rest mutate the game state in place by playing moves. If we didn't copy, the second iteration would start from wherever the first one ended. The deepcopy is the slowest single step per iteration — about 60-70% of total time — and is the obvious target if we ever optimize.

---

## `best_move(root)`

**What it does:** returns the most-visited child's move, or `None` if `root` has no children.

| Input | Output |
|---|---|
| `root` with at least one child | `(r, c)` of the child with highest `visits` |
| `root` with no children | `None` |

**Why most-visited and not highest win rate:** UCB1 already pulls visits toward children with high win rates, so the most-visited child usually IS the highest-win-rate child. But "most visits" is more robust to noise — a child with 1 visit and a 100% win rate looks great by win-rate but is essentially untested.

---

## `class MCTSAgent` (in `ai/agent.py`)

**What it is:** the GUI-facing wrapper around `search`. Owns a persistent root Node so it can reuse search work between turns. This is the only class the GUI and tests should touch.

### `__init__(iterations=2000, c=sqrt(2), seed=None)`

| Parameter | Meaning |
|---|---|
| `iterations` | how many MCTS iterations per move (more = stronger but slower) |
| `c` | UCB1 exploration constant; default sqrt(2) |
| `seed` | RNG seed for reproducibility (pass `None` for non-deterministic play) |

### `reset()`

**What it does:** clears the stored search tree. Call this when starting a new game.

### `select_move(game)`

**What it does:** returns the agent's chosen move for `game`'s current player. Either an `(r, c)` tuple or the string `"pass"`.

| Input | Output |
|---|---|
| Position with legal non-eye moves | `(r, c)` |
| Position where every empty is own-eye OR every candidate turns out illegal | `"pass"` |

**Algorithm:**

1. If the cheap candidate filter says no moves → return `"pass"`.
2. Sync the stored root to the current game state (descend into the opponent's move if we explored it; otherwise build a fresh root).
3. Run `search` for `iterations` iterations from the root.
4. Pick `best_move`. If it's `None` (every candidate was illegal), return `"pass"`.
5. Promote the chosen child to be the new root for the next call.
6. Return the move.

### `_advance_root_to(game)` (private)

**What it does:** keeps the stored search tree aligned with the actual game.

When the agent's previous `select_move` finished, we promoted the chosen child to root — that root represents the state **after our move**. Now the opponent has played `game.last_move`. If we already explored that exact reply (it's in `_root.children`), we descend into it and inherit all its accumulated visits/wins. If we didn't, we throw away the tree and start fresh from the current game.

| Input | Effect |
|---|---|
| `game` whose `last_move` matches a child in our tree | descend; new root has the inherited visits |
| `game` whose `last_move` is something we never explored | discard tree; build fresh root |
| First call after `reset()` | build fresh root |

### `_promote(move)` (private)

**What it does:** makes `_root.children[move]` the new root and detaches its parent pointer. One-line helper used by both selection and root-syncing.

---

## Tree reuse, in slow motion

A common point of confusion: how does the tree survive between calls? Walk through what happens across two turns:

1. **First call.** Build root from current game. Run search. UCB1 spends visits across all 80-ish candidate moves, expanding the tree several plies deep for promising ones. Pick the most-visited child as our move (call it `M_us`). Promote it: `_root` is now the post-our-move state, with `_root.children` containing every opponent reply we explored.
2. Player makes `M_us` on the actual board. Then the opponent plays `M_opp`.
3. **Second call.** `game.last_move` is now `M_opp`. Look up `_root.children[M_opp]` — if present, we explored that reply during the previous search, including its subtree. Promote it: `_root` is now the state after both moves, **with all the visits/wins from the previous search still intact**. Run search for another `iterations` iterations on top.

The result: 2000 iterations on call 1 + 2000 on call 2 ≠ 2000 of useful work — call 2 starts with maybe 100-300 visits already in the relevant subtree from call 1.

---

## Performance notes

Measured on the dev machine, default settings, fresh empty board:

| Iterations | Time per move |
|---|---|
| 200 | ~0.6s |
| 400 | ~1.3s |
| 800 | ~3.5s |
| 2000 | ~9s |

Bottleneck is `copy.deepcopy(root_game)` per iteration. The obvious optimization would be to write a custom snapshot/restore routine that copies just the flat board list (cheap) instead of the whole `GoGame` object — left as a Day 4 buffer item.

The cheap `candidate_moves` (no `is_legal` calls) was already a 10× speedup over the obvious "filter to legal moves first" implementation; that change is locked in.

---

## Why this design is small on purpose

- **One source of truth for legality.** The cheap filter feeds `_expand` and `_simulate`; both delegate the actual legality check to `place_stone`'s return value. No duplicated rule code.
- **No tunable thresholds, no priors, no RAVE.** Pure UCB1 with random rollouts and the eye heuristic. Easier to defend in the oral; easier to debug; well-studied baseline.
- **Tree reuse without complex sync logic.** We trust `game.last_move` and a single dict lookup. If the lookup misses, we throw the tree away — correctness is preserved at the cost of one slower turn.
- **Pass policy hard-coded to the spec.** Agent only ever passes when forced to. No exploration of pass-as-a-strategy, because in this assignment passing always loses.
