"""Quick MCTS speed benchmark (no GUI).

Measures:
- average AI move latency (ms)
- total AI moves evaluated
- effective iterations/sec

Run from repo root:
    python3 scripts/benchmark_mcts.py --iterations 400 --games 3
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.agent import MCTSAgent
from engine.go_engine import BLACK, GoGame


def run_one_game(iterations: int, seed: int, max_moves: int) -> tuple[list[float], int, bool]:
    """Return per-move latencies, move count, and finished flag."""
    game = GoGame()
    black = MCTSAgent(iterations=iterations, seed=seed)
    white = MCTSAgent(iterations=iterations, seed=seed + 1)

    latencies: list[float] = []
    moves = 0
    while not game.finished and moves < max_moves:
        agent = black if game.to_move == BLACK else white
        t0 = time.perf_counter()
        move = agent.select_move(game)
        latencies.append(time.perf_counter() - t0)

        if move == "pass":
            game.pass_turn()
        else:
            ok = game.place_stone(*move)
            if not ok:
                raise RuntimeError(f"Agent returned illegal move: {move}")
        moves += 1

    return latencies, moves, game.finished


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark MCTS speed on self-play games")
    p.add_argument("--iterations", type=int, default=400, help="MCTS iterations per move")
    p.add_argument("--games", type=int, default=3, help="Number of self-play games")
    p.add_argument("--max-moves", type=int, default=200, help="Safety cap per game")
    p.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = p.parse_args()

    all_latencies: list[float] = []
    total_moves = 0
    finished_games = 0

    bench_start = time.perf_counter()
    for i in range(args.games):
        latencies, moves, finished = run_one_game(
            iterations=args.iterations,
            seed=args.seed + 1000 * i,
            max_moves=args.max_moves,
        )
        all_latencies.extend(latencies)
        total_moves += moves
        if finished:
            finished_games += 1
    elapsed = time.perf_counter() - bench_start

    if not all_latencies:
        print("No moves played; benchmark inconclusive.")
        return

    avg_ms = statistics.mean(all_latencies) * 1000.0
    p50_ms = statistics.median(all_latencies) * 1000.0
    p95_ms = statistics.quantiles(all_latencies, n=20)[18] * 1000.0 if len(all_latencies) >= 20 else max(all_latencies) * 1000.0
    iters_per_sec = (total_moves * args.iterations) / elapsed if elapsed > 0 else 0.0

    print(f"Games: {args.games} (finished: {finished_games})")
    print(f"Iterations/move: {args.iterations}")
    print(f"Total moves: {total_moves}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Avg move latency: {avg_ms:.2f} ms")
    print(f"P50 latency: {p50_ms:.2f} ms")
    print(f"P95 latency: {p95_ms:.2f} ms")
    print(f"Effective throughput: {iters_per_sec:.1f} iterations/sec")


if __name__ == "__main__":
    main()
