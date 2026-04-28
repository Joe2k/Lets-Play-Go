"""Head-to-head evaluation for MCTS / PUCT agent configs.

Runs multiple games between two agents (alternating colors) and reports:
- wins / win rate
- average score margin
- move latency statistics

Use --a-engine / --b-engine to mix MCTS and PUCT, e.g.:

    # PUCT(net_new) vs PUCT(net_old) — promotion gate.
    python3 scripts/eval_match.py --games 40 \\
        --a-name new --a-engine puct --a-model checkpoints/pv_new.pt --a-iters 200 \\
        --b-name old --b-engine puct --b-model checkpoints/pv_old.pt --b-iters 200

    # PUCT(net) vs MCTS baseline.
    python3 scripts/eval_match.py --games 40 \\
        --a-name puct --a-engine puct --a-model checkpoints/pv_latest.pt --a-iters 200 \\
        --b-name mcts --b-engine mcts --b-iters 400
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.agent import MCTSAgent
from engine.go_engine import BLACK, WHITE, GoGame


@dataclass
class AgentSpec:
    name: str
    iterations: int
    engine: str = "mcts"
    model_path: Optional[str] = None


def _build_agent(spec: AgentSpec, seed: int):
    if spec.engine == "puct":
        from ai.puct_agent import PUCTAgent
        return PUCTAgent(
            iterations=spec.iterations,
            model_path=spec.model_path,
            seed=seed,
        )
    return MCTSAgent(iterations=spec.iterations, seed=seed)


@dataclass
class GameResult:
    black_name: str
    white_name: str
    winner_name: str
    black_score: float
    white_score: float
    margin_for_black: float
    moves: int
    finished: bool
    black_avg_ms: float
    white_avg_ms: float


def _play_one_game(
    black_spec: AgentSpec,
    white_spec: AgentSpec,
    seed: int,
    max_moves: int,
) -> GameResult:
    game = GoGame()
    black_agent = _build_agent(black_spec, seed)
    white_agent = _build_agent(white_spec, seed + 1)

    black_latencies: list[float] = []
    white_latencies: list[float] = []
    move_count = 0

    while not game.finished and move_count < max_moves:
        is_black_turn = game.to_move == BLACK
        agent = black_agent if is_black_turn else white_agent
        t0 = time.perf_counter()
        move = agent.select_move(game)
        dt = (time.perf_counter() - t0) * 1000.0
        if is_black_turn:
            black_latencies.append(dt)
        else:
            white_latencies.append(dt)

        if move == "pass":
            game.pass_turn()
        else:
            ok = game.place_stone(*move)
            if not ok:
                raise RuntimeError(
                    f"Illegal move from {black_spec.name if is_black_turn else white_spec.name}: {move}"
                )
        move_count += 1

    s = game.score()
    winner_name = black_spec.name if s["winner"] == BLACK else white_spec.name
    return GameResult(
        black_name=black_spec.name,
        white_name=white_spec.name,
        winner_name=winner_name,
        black_score=s["black"],
        white_score=s["white"],
        margin_for_black=s["black"] - s["white"],
        moves=move_count,
        finished=game.finished,
        black_avg_ms=(statistics.mean(black_latencies) if black_latencies else 0.0),
        white_avg_ms=(statistics.mean(white_latencies) if white_latencies else 0.0),
    )


def _append_csv(csv_path: pathlib.Path, rows: list[GameResult]) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "black_name",
                    "white_name",
                    "winner_name",
                    "black_score",
                    "white_score",
                    "margin_for_black",
                    "moves",
                    "finished",
                    "black_avg_ms",
                    "white_avg_ms",
                ]
            )
        for r in rows:
            w.writerow(
                [
                    r.black_name,
                    r.white_name,
                    r.winner_name,
                    f"{r.black_score:.2f}",
                    f"{r.white_score:.2f}",
                    f"{r.margin_for_black:.2f}",
                    r.moves,
                    int(r.finished),
                    f"{r.black_avg_ms:.2f}",
                    f"{r.white_avg_ms:.2f}",
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate two agent configs head-to-head")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--max-moves", type=int, default=220)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.add_argument("--a-iters", type=int, default=400)
    p.add_argument("--b-iters", type=int, default=400)
    p.add_argument("--a-engine", choices=["mcts", "puct"], default="mcts")
    p.add_argument("--b-engine", choices=["mcts", "puct"], default="mcts")
    p.add_argument("--a-model", type=str, default=None)
    p.add_argument("--b-model", type=str, default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--gate-threshold", type=float, default=None,
                   help="If set, exit code 0 only when A's win rate >= this value (e.g. 0.55).")
    args = p.parse_args()

    a = AgentSpec(name=args.a_name, iterations=args.a_iters,
                  engine=args.a_engine, model_path=args.a_model)
    b = AgentSpec(name=args.b_name, iterations=args.b_iters,
                  engine=args.b_engine, model_path=args.b_model)
    results: list[GameResult] = []

    started = time.perf_counter()
    for i in range(args.games):
        if i % 2 == 0:
            black_spec, white_spec = a, b
        else:
            black_spec, white_spec = b, a
        results.append(
            _play_one_game(
                black_spec=black_spec,
                white_spec=white_spec,
                seed=args.seed + 1009 * i,
                max_moves=args.max_moves,
            )
        )
    elapsed = time.perf_counter() - started

    a_wins = sum(1 for r in results if r.winner_name == a.name)
    b_wins = len(results) - a_wins
    a_winrate = a_wins / len(results) if results else 0.0
    avg_moves = statistics.mean(r.moves for r in results) if results else 0.0
    avg_margin_for_black = statistics.mean(r.margin_for_black for r in results) if results else 0.0

    margins_for_a: list[float] = []
    for r in results:
        if r.black_name == a.name:
            margins_for_a.append(r.margin_for_black)
        else:
            margins_for_a.append(-r.margin_for_black)
    avg_margin_for_a = statistics.mean(margins_for_a) if margins_for_a else 0.0

    a_move_latencies: list[float] = []
    b_move_latencies: list[float] = []
    for r in results:
        if r.black_name == a.name:
            a_move_latencies.append(r.black_avg_ms)
            b_move_latencies.append(r.white_avg_ms)
        else:
            a_move_latencies.append(r.white_avg_ms)
            b_move_latencies.append(r.black_avg_ms)

    print(f"Games: {len(results)} in {elapsed:.2f}s")
    print(f"{a.name} ({a.engine}, iters={a.iterations}) wins: {a_wins} ({a_winrate*100:.1f}%)")
    print(f"{b.name} ({b.engine}, iters={b.iterations}) wins: {b_wins} ({(1-a_winrate)*100:.1f}%)")
    print(f"Avg moves/game: {avg_moves:.1f}")
    print(f"Avg score margin for Black: {avg_margin_for_black:+.2f}")
    print(f"Avg score margin for {a.name}: {avg_margin_for_a:+.2f}")
    print(f"{a.name} avg move latency: {statistics.mean(a_move_latencies):.2f} ms")
    print(f"{b.name} avg move latency: {statistics.mean(b_move_latencies):.2f} ms")

    if args.csv:
        csv_path = pathlib.Path(args.csv)
        _append_csv(csv_path, results)
        print(f"Saved per-game results to {csv_path}")

    if args.gate_threshold is not None:
        passed = a_winrate >= args.gate_threshold
        print(f"Gate threshold {args.gate_threshold:.2f}: {'PASS' if passed else 'FAIL'}")
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
