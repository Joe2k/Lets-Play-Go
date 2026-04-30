"""Generate self-play training data using PUCT visit targets.

Outputs a torch checkpoint dict with:
- states: [N, 3, 9, 9] float tensor
- policy: [N, 81] float tensor
- value:  [N] float tensor in {-1, +1}
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import queue
import random
import sys
import time
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.model import PolicyValueModel, encode_game_tensor, require_torch, torch
from ai.puct_agent import PUCTNode, run_puct_search, run_batched_puct_search
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
    if checkpoint_path:
        _worker_predictor = PolicyValueModel(model_path=checkpoint_path, device=device)
    else:
        _worker_predictor = _UniformPredictor()


def _play_batch_games(
    start_game_idx: int,
    num_games: int,
    batch_size: int,
    seed_base: int,
    max_moves: int,
    iterations: int,
    temperature: float,
    add_root_noise: bool,
    c_puct: float = 1.25,
    progress_queue: Optional[mp.Queue] = None,
    predictor: Optional[PolicyValueModel] = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]]:
    """Plays multiple games using batched MCTS and returns a list of game results."""
    results = []
    
    active_games: list[GoGame] = []
    active_roots: list[Optional[PUCTNode]] = []
    active_seeds: list[int] = []
    active_indices: list[int] = []
    
    game_states: list[list[torch.Tensor]] = []
    game_policies: list[list[torch.Tensor]] = []
    game_players: list[list[int]] = []
    
    next_idx_to_start = 0
    
    current_predictor = predictor or _worker_predictor
    
    def start_game():
        nonlocal next_idx_to_start
        if next_idx_to_start >= num_games:
            return False
        
        g_idx = start_game_idx + next_idx_to_start
        seed = seed_base + 10007 * g_idx
        active_games.append(GoGame())
        active_roots.append(None)
        active_seeds.append(seed)
        active_indices.append(g_idx)
        
        game_states.append([])
        game_policies.append([])
        game_players.append([])
        
        next_idx_to_start += 1
        return True

    for _ in range(min(batch_size, num_games)):
        start_game()
        
    plies_simulated = 0
    while active_games:
        plies_simulated += 1
        if plies_simulated % 10 == 0 and progress_queue is None:
            print(f"  [batch progress] simulating move {plies_simulated} for {len(active_games)} active games...", flush=True)
            
        roots_to_search = []
        for i in range(len(active_games)):
            if active_roots[i] is None:
                active_roots[i] = PUCTNode(to_play=active_games[i].to_move, prior=1.0)
            roots_to_search.append(active_roots[i])
            
        if isinstance(current_predictor, PolicyValueModel):
            run_batched_puct_search(
                games=active_games,
                model=current_predictor,
                iterations=iterations,
                c_puct=c_puct,
                roots=roots_to_search,
                add_root_noise=add_root_noise,
                rng=random.Random(active_seeds[0] if active_seeds else 0),
            )
        else:
            for i in range(len(active_games)):
                run_puct_search(
                    game=active_games[i],
                    predictor=current_predictor,
                    iterations=iterations,
                    c_puct=c_puct,
                    root=roots_to_search[i],
                    add_root_noise=add_root_noise,
                    rng=random.Random(active_seeds[i]),
                )
        
        indices_to_remove = []
        for i in range(len(active_games)):
            game = active_games[i]
            root = roots_to_search[i]
            g_idx = active_indices[i]
            
            visit_sum = sum(ch.visit_count for ch in root.children.values())
            if visit_sum <= 0 or not root.children or game.finished or len(game_states[i]) >= max_moves:
                winner = game.score()["winner"]
                p_list = game_players[i]
                v_list = [1.0 if p == winner else -1.0 for p in p_list]
                
                st_tensor = torch.stack(game_states[i]) if game_states[i] else torch.empty((0, 3, 9, 9), dtype=torch.float32)
                pol_tensor = torch.stack(game_policies[i]) if game_policies[i] else torch.empty((0, 81), dtype=torch.float32)
                val_tensor = torch.tensor(v_list, dtype=torch.float32) if v_list else torch.empty((0,), dtype=torch.float32)
                
                res = (st_tensor, pol_tensor, val_tensor, len(p_list), winner, g_idx)
                results.append(res)
                if progress_queue is not None:
                    progress_queue.put(res)
                else:
                    # In single-worker mode without a queue, print progress immediately
                    print(f"  -> Game {g_idx} finished in {len(p_list)} moves (winner: {winner})", flush=True)
                indices_to_remove.append(i)
                continue
                
            pol = torch.zeros(81, dtype=torch.float32)
            visits = {}
            for move, ch in root.children.items():
                v = ch.visit_count
                visits[move] = v
                pol[move[0] * 9 + move[1]] = float(v) / float(visit_sum)
                
            st = encode_game_tensor(game)[0].cpu()
            game_states[i].append(st)
            game_policies[i].append(pol)
            game_players[i].append(game.to_move)
            
            rng = random.Random(active_seeds[i] + len(game_states[i]))
            temp = temperature if len(game_states[i]) < 20 else max(0.05, temperature * 0.25)
            mv = _sample_from_visits(visits, rng, temp)
            
            if not game.place_stone(*mv):
                game.pass_turn()
                active_roots[i] = None
            else:
                if mv in root.children:
                    active_roots[i] = root.children[mv]
                else:
                    active_roots[i] = None
                
        for idx in reversed(indices_to_remove):
            active_games.pop(idx)
            active_roots.pop(idx)
            active_seeds.pop(idx)
            active_indices.pop(idx)
            game_states.pop(idx)
            game_policies.pop(idx)
            game_players.pop(idx)
            start_game()
            
    return results


def main() -> None:
    require_torch()
    p = argparse.ArgumentParser(description="Generate PUCT self-play dataset")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--max-moves", type=int, default=220)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="data/selfplay.pt")
    p.add_argument("--predictor-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--c-puct", type=float, default=1.25,
                   help="Exploration constant for PUCT search.")
    args = p.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    states, policies, values = [], [], []

    def _flush(reason: str) -> None:
        blob = {
            "states": torch.cat(states) if states else torch.empty((0, 3, 9, 9), dtype=torch.float32),
            "policy": torch.cat(policies) if policies else torch.empty((0, 81), dtype=torch.float32),
            "value": torch.cat(values) if values else torch.empty((0,), dtype=torch.float32),
        }
        torch.save(blob, out_path)
        print(f"[{reason}] saved {blob['states'].shape[0]} samples to {out_path}", flush=True)

    started = time.perf_counter()
    games_completed = 0
    
    if args.workers == 1:
        print(f"Starting 1 worker on {args.device} (batch_size={args.batch_size})...", flush=True)
        predictor = None
        if args.predictor_checkpoint:
            predictor = PolicyValueModel(model_path=args.predictor_checkpoint, device=args.device)
        else:
            predictor = _UniformPredictor()
            
        # We can just run one big batch if it's 1 worker
        results = _play_batch_games(
            start_game_idx=0,
            num_games=args.games,
            batch_size=args.batch_size,
            seed_base=args.seed,
            max_moves=args.max_moves,
            iterations=args.iterations,
            temperature=args.temperature,
            add_root_noise=not args.no_noise,
            c_puct=args.c_puct,
            progress_queue=None,
            predictor=predictor,
        )
        for st, pol, val, moves, winner, g_idx in results:
            states.append(st)
            policies.append(pol)
            values.append(val)
            games_completed += 1
            elapsed = time.perf_counter() - started
            eta = (args.games - games_completed) * (elapsed / games_completed)
            print(f"[{games_completed}/{args.games}] game={g_idx} samples={moves:3d} | "
                  f"winner={winner} | total={elapsed:6.1f}s | eta={eta:5.1f}s", flush=True)
            if args.save_every > 0 and games_completed % args.save_every == 0:
                _flush("checkpoint")
    else:
        # Multi-worker mode
        games_per_worker = args.games // args.workers
        remainder = args.games % args.workers
        worker_tasks = []
        current_start = 0
        for i in range(args.workers):
            num = games_per_worker + (1 if i < remainder else 0)
            if num > 0:
                worker_tasks.append((
                    current_start,
                    num,
                    args.batch_size,
                    args.seed,
                    args.max_moves,
                    args.iterations,
                    args.temperature,
                    not args.no_noise,
                    args.c_puct,
                ))
                current_start += num

        print(f"Starting {args.workers} workers on {args.device} (batch_size={args.batch_size})...", flush=True)
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        progress_queue = manager.Queue()
        
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(args.predictor_checkpoint, args.device),
        )
        
        futs = [pool.submit(_play_batch_games, *t, progress_queue=progress_queue) for t in worker_tasks]
        
        try:
            while games_completed < args.games:
                try:
                    res = progress_queue.get(timeout=1.0)
                    st, pol, val, moves, winner, g_idx = res
                    states.append(st)
                    policies.append(pol)
                    values.append(val)
                    games_completed += 1
                    elapsed = time.perf_counter() - started
                    eta = (args.games - games_completed) * (elapsed / games_completed)
                    print(f"[{games_completed}/{args.games}] game={g_idx} samples={moves:3d} | "
                          f"winner={winner} | total={elapsed:6.1f}s | eta={eta:5.1f}s", flush=True)
                    if args.save_every > 0 and games_completed % args.save_every == 0:
                        _flush("checkpoint")
                except queue.Empty:
                    # This happens when timeout=1.0 is reached
                    pass
                except Exception as e:
                    print(f"Error reading from queue: {e}")
                
                # Check for completed or crashed workers
                if all(f.done() for f in futs) and progress_queue.empty():
                    for idx, f in enumerate(futs):
                        if f.exception() is not None:
                            print(f"\n[ERROR] Worker {idx} crashed with exception:\n{f.exception()}", file=sys.stderr)
                    break
        except KeyboardInterrupt:
            print("\nInterrupted.")
            _flush("interrupted")
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)
        pool.shutdown(wait=False, cancel_futures=True)

    _flush("done")


if __name__ == "__main__":
    main()
