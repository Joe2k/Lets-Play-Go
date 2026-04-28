"""Generate self-play training data using PUCT visit targets.

Outputs a torch checkpoint dict with:
- states: [N, 3, 9, 9] float tensor
- policy: [N, 81] float tensor
- value:  [N] float tensor in {-1, +1}

Use --predictor-checkpoint to drive self-play with a trained policy-value
network instead of the uniform bootstrap predictor. Dirichlet root noise
is added during data generation to keep openings diverse.
"""

from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import pathlib
import random
import sys
import time
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.model import PolicyValueModel, encode_game_tensor, require_torch, torch
from ai.puct_agent import PUCTNode, run_puct_search
from engine.go_engine import GoGame


class _UniformPredictor:
    """Uniform prior + zero value bootstrap predictor."""

    def predict(self, game: GoGame) -> tuple[list[float], float]:
        return [1.0 / 81.0] * 81, 0.0


def _sample_from_visits(visits: dict[tuple[int, int], int], rng: random.Random, temperature: float) -> tuple[int, int]:
    items = list(visits.items())
    if temperature <= 1e-6:
        return max(items, key=lambda kv: kv[1])[0]
    weights = [max(1e-9, float(v) ** (1.0 / temperature)) for _, v in items]
    moves = [m for m, _ in items]
    return rng.choices(moves, weights=weights, k=1)[0]


# --- Global worker state ---
_worker_predictor = None


def _init_worker(checkpoint_path: Optional[str], device: str) -> None:
    global _worker_predictor
    # Each worker initializes its own model on spawn/fork to avoid IPC/MPS context issues.
    if checkpoint_path:
        _worker_predictor = PolicyValueModel(model_path=checkpoint_path, device=device)
    else:
        _worker_predictor = _UniformPredictor()


def _play_one_game(
    game_idx: int,
    seed: int,
    max_moves: int,
    iterations: int,
    temperature: float,
    add_root_noise: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Runs a single self-play game and returns (states, policy, value, move_count, winner)."""
    rng = random.Random(seed)
    game = GoGame()

    states_list = []
    policies_list = []
    player_to_move_list = []

    root: Optional[PUCTNode] = None

    for ply in range(max_moves):
        if game.finished:
            break

        # Tree reuse: if we have a root, run_puct_search will use its subtree.
        root = run_puct_search(
            game=game,
            predictor=_worker_predictor,
            iterations=iterations,
            c_puct=1.4,
            root=root,
            add_root_noise=add_root_noise,
            rng=rng,
        )

        if not root.children:
            game.pass_turn()
            break

        visit_sum = sum(ch.visit_count for ch in root.children.values())
        if visit_sum <= 0:
            game.pass_turn()
            break

        pol = torch.zeros(81, dtype=torch.float32)
        visits: dict[tuple[int, int], int] = {}
        for move, ch in root.children.items():
            v = ch.visit_count
            visits[move] = v
            pol[move[0] * 9 + move[1]] = float(v) / float(visit_sum)

        st = encode_game_tensor(game)[0].cpu()
        states_list.append(st)
        policies_list.append(pol)
        player_to_move_list.append(game.to_move)

        temp = temperature if ply < 20 else max(0.05, temperature * 0.25)
        mv = _sample_from_visits(visits, rng=rng, temperature=temp)

        if not game.place_stone(*mv):
            game.pass_turn()
            break

        # Tree reuse: promote chosen child to new root for the next ply.
        if mv in root.children:
            root = root.children[mv]
        else:
            root = None

    winner = game.score()["winner"]
    values_list = []
    for p in player_to_move_list:
        values_list.append(1.0 if p == winner else -1.0)

    st_tensor = torch.stack(states_list) if states_list else torch.empty((0, 3, 9, 9), dtype=torch.float32)
    pol_tensor = torch.stack(policies_list) if policies_list else torch.empty((0, 81), dtype=torch.float32)
    val_tensor = torch.tensor(values_list, dtype=torch.float32) if values_list else torch.empty((0,), dtype=torch.float32)

    return st_tensor, pol_tensor, val_tensor, len(player_to_move_list), winner


def _play_one_game_wrapper(args_tuple):
    return _play_one_game(*args_tuple)


def main() -> None:
    require_torch()
    p = argparse.ArgumentParser(description="Generate PUCT self-play dataset")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--max-moves", type=int, default=220)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="data/selfplay.pt")
    p.add_argument("--predictor-checkpoint", type=str, default=None,
                   help="Optional path to a trained policy-value checkpoint. "
                        "If unset, uses a uniform bootstrap predictor.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-noise", action="store_true",
                   help="Disable Dirichlet root noise (debugging only).")
    p.add_argument("--save-every", type=int, default=10,
                   help="Flush partial dataset to --output every N games (Ctrl+C-safe).")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes.")
    args = p.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    states = []
    policies = []
    values = []

    def _flush(reason: str) -> None:
        blob = {
            "states": torch.cat(states) if states else torch.empty((0, 3, 9, 9), dtype=torch.float32),
            "policy": torch.cat(policies) if policies else torch.empty((0, 81), dtype=torch.float32),
            "value": torch.cat(values) if values else torch.empty((0,), dtype=torch.float32),
        }
        torch.save(blob, out_path)
        print(f"[{reason}] saved {blob['states'].shape[0]} samples to {out_path}", flush=True)

    add_root_noise = not args.no_noise
    started = time.perf_counter()
    games_completed = 0

    ctx = mp.get_context("spawn")
    print(f"Starting {args.workers} workers on {args.device}...", flush=True)
    try:
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.predictor_checkpoint, args.device)
        ) as pool:
            # Prepare task seeds
            seeds = [args.seed + 10007 * i for i in range(args.games)]
            task_args = [
                (i, s, args.max_moves, args.iterations, args.temperature, add_root_noise)
                for i, s in enumerate(seeds)
            ]

            # imap_unordered yields results as soon as any worker finishes
            for res in pool.imap_unordered(_play_one_game_wrapper, task_args):
                st, pol, val, moves, winner = res
                states.append(st)
                policies.append(pol)
                values.append(val)
                games_completed += 1

                total_elapsed = time.perf_counter() - started
                games_per_sec = games_completed / total_elapsed
                eta_seconds = (args.games - games_completed) / games_per_sec if games_per_sec > 0 else 0
                
                print(
                    f"[{games_completed}/{args.games}] samples={moves:3d} | "
                    f"winner={winner} | total={total_elapsed:6.1f}s | "
                    f"eta={eta_seconds:6.1f}s",
                    flush=True
                )

                if args.save_every > 0 and games_completed % args.save_every == 0:
                    _flush("checkpoint")
    except KeyboardInterrupt:
        print("\nInterrupted — flushing partial dataset.")
        _flush("interrupted")
        sys.exit(0)

    _flush("done")


if __name__ == "__main__":
    main()
