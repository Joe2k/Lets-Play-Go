# MCTS Test Guide

Plain-English walkthrough of every test in [tests/test_mcts.py](../tests/test_mcts.py). Each test is a sanity check on one aspect of `MCTSAgent` — these are not strength benchmarks, just guards that pin down the agent's contract.

If you don't know the underlying terms (eye, candidate move, tree reuse), read [GLOSSARY.md](GLOSSARY.md) and [MCTS.md](MCTS.md) first.

---

## 1. `test_does_not_pass_when_moves_available`

**What it checks:** on a fresh empty board, the agent picks an actual coordinate, not `"pass"`.

**Why it matters:** in this project, passing = conceding the game. An agent that passes when it has good moves available would lose every game on move 1. This test catches that whole class of bug.

| Input | Expected |
|---|---|
| `MCTSAgent(iterations=50, seed=0).select_move(GoGame())` | a `(row, col)` tuple with `0 ≤ row, col < 9` |

---

## 2. `test_does_not_consider_filling_own_eye`

**What it checks:** the agent's "don't fill your own eyes" filter actually filters. Sets up a Black ring around a single empty point at `(2, 2)` and verifies (a) `candidate_moves` excludes `(2, 2)`, (b) the agent never returns `(2, 2)`.

Setup (Black wall around a single own-eye):

```
. . . . . . . . .
. B B B . . . . .
. B . B . . . . .
. B B B . . . . .
. . . . . . . . .
...
```

| Input | Expected |
|---|---|
| `candidate_moves(g)` | does not contain `(2, 2)` |
| `MCTSAgent(iterations=100, seed=0).select_move(g)` | not equal to `(2, 2)` |

**Why it matters:** filling your own eye is the easiest way for random-rollout MCTS to suicide a winning group. This is the single heuristic that makes random rollouts useful in Go, so it has to actually work.

---

## 3. `test_passes_when_only_own_eyes_remain`

**What it checks:** when there are zero candidate moves (every empty point is the agent's own eye), `select_move` returns `"pass"` instead of returning a tuple.

Setup (board entirely Black except for two own-eye points at `(4,3)` and `(4,5)`):

```
B B B B B B B B B
B B B B B B B B B
B B B B B B B B B
B B B B B B B B B
B B B . B . B B B
B B B B B B B B B
...
```

| Input | Expected |
|---|---|
| `candidate_moves(g)` | `[]` (empty list) |
| `MCTSAgent(iterations=10, seed=0).select_move(g)` | `"pass"` |

**Why it matters:** the agent must be able to recognize "no moves left" and pass. Otherwise `select_move` would either crash or loop forever trying to find a move that doesn't exist.

---

## 4. `test_self_play_game_completes_without_crash`

**What it checks:** two MCTS agents play each other from an empty board, with low iterations for speed. The game must terminate (either by reaching a pass or hitting the 200-move safety cap), every move must be either `"pass"` or a legal coordinate, and `score()` must return a valid winner at the end.

| Step | Behavior |
|---|---|
| Loop while not finished and `move_count < 200` | each agent's `select_move` returns a move |
| If move is `"pass"` | `g.pass_turn()` (game ends) |
| Otherwise | `g.place_stone(*move)` must return `True` (move is legal) |
| After loop | `g.finished is True` and `g.score()["winner"]` is `BLACK` or `WHITE` |

**Why it matters:** this is the integration test. It exercises every path in MCTS at least once: tree reuse across turns, eye filtering, suicide handling, normal play, and end-of-game pass. If the agent has a bug that only shows up in actual play (not in the targeted unit tests), this is the test most likely to catch it.

This replaced an earlier "MCTS picks the obvious capture" test that turned out flaky — in the test position random rollouts captured the threatened group eventually regardless of root move, so MCTS couldn't see the immediate capture as uniquely better. Strength tests need carefully constructed positions where root-level move order changes the rollout outcome; that's a benchmark, not a sanity check.

---

## 5. `test_tree_reuse_descends_into_existing_subtree`

**What it checks:** after the first `select_move` returns, the agent's stored root contains explored opponent replies. If we play one of those replies on the actual board and then call `_advance_root_to`, the new root carries the visit count from the previous search — proving the subtree was inherited rather than rebuilt from scratch.

| Step | Action | Expected |
|---|---|---|
| 1 | `select_move(g)` with 300 iterations | returns a move; `agent._root.children` has multiple explored opponent replies |
| 2 | Play our move on `g`; pick the opponent reply with the most inherited visits | `inherited_visits > 0` |
| 3 | Play the opponent reply on `g`; call `agent._advance_root_to(g)` | `agent._root.visits == inherited_visits` (exactly equal — no fresh search has run yet) |

**Why it matters:** tree reuse is the most error-prone part of the agent. If `_advance_root_to` ever falls through to building a fresh root when it shouldn't (e.g., because of a state-comparison bug), we'd silently waste all the search work from the previous turn. This test catches that by checking the visits field directly.

This test deliberately reaches into the agent's internals (`_root`, `_advance_root_to`). Internal-state testing is fine for a feature this fragile.

---

## 6. `test_seed_determinism`

**What it checks:** two agents with the same seed, given the same starting position, produce the same move.

| Input | Expected |
|---|---|
| `a1 = MCTSAgent(iterations=100, seed=42)`, `a2 = MCTSAgent(iterations=100, seed=42)` | `a1.select_move(GoGame()) == a2.select_move(GoGame())` |

**Why it matters:** without determinism, every other test in this file would be flaky — you could never tell whether a failure was a real regression or just bad luck on a particular RNG roll. The agent threads its own `random.Random(seed)` through all the random choices (rollouts, expansion order) precisely so this test can pin behavior down.

---

## What these tests deliberately don't cover

- **Strength.** "Does MCTS beat a uniformly random player?" or "does it pick optimal moves in tactical positions?" are benchmark questions, not sanity questions. They need carefully designed positions and many iterations to produce stable answers, and they belong in a separate benchmark suite if at all.
- **Performance.** Iteration counts are kept small (10-300) so the suite runs in under 10 seconds. The actual perf characteristics are documented in [MCTS.md](MCTS.md).
- **Specific UCB1 numerics.** UCB1 values are floating-point and hard to assert against exactly. We trust the algorithm; we test the surrounding scaffolding.

---

## Running the tests

From the repo root with the project's `.venv` activated:

```bash
pytest -v tests/test_mcts.py
```

All 6 tests should pass in under 10 seconds. To run them alongside the engine tests:

```bash
pytest -v tests/
```

That's 23 tests total (17 engine + 6 MCTS). If one fails, the test name tells you which contract regressed.
