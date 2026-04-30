"""Generate self-play training data using PUCT visit targets.

Outputs a torch checkpoint dict with:
- states: [N, INPUT_PLANES, 9, 9] float tensor
- policy: [N, POLICY_SIZE] float tensor (board moves + pass slot)
- value:  [N] float tensor in {-1, +1}
- ownership: [N, 81] float tensor in {-1, +1} from side-to-move perspective
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import pathlib
import queue
import random
import sys
import time
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "1"

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.model import (
    INPUT_PLANES,
    PASS_INDEX,
    POLICY_SIZE,
    PolicyValueModel,
    encode_game_tensor,
    require_torch,
    torch,
)
from ai.puct_agent import PASS_MOVE, PUCTNode, _sample_from_visits, run_puct_search, run_batched_puct_search
from ai.mcts import Node as MCTSNode, best_move as mcts_best_move, candidate_moves, search as mcts_search
from engine.go_engine import BLACK, SIZE, WHITE, GoGame


class _UniformPredictor:
    """Uniform prior + zero value bootstrap predictor."""

    def predict(self, game: GoGame) -> tuple[list[float], float]:
        return [1.0 / POLICY_SIZE] * POLICY_SIZE, 0.0


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
    fast_iterations: int = 10,
    full_search_fraction: float = 0.25,
    pass_penalty: float = 0.0,
    resign_threshold: float = 0.9,
    resign_min_moves: int = 30,
    value_margin_scale: float = 20.0,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]]:
    """Plays multiple games using batched MCTS and returns a list of game results."""
    results = []

    active_games: list[GoGame] = []
    active_roots: list[Optional[PUCTNode]] = []
    active_seeds: list[int] = []
    active_indices: list[int] = []

    game_states: list[list[torch.Tensor]] = []
    game_policies: list[list[torch.Tensor]] = []
    game_players: list[list[int]] = []
    game_ownerships: list[list[torch.Tensor]] = []

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
        game_ownerships.append([])

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

        # Playout Cap Randomization (PCR): per-turn full vs fast search
        is_full_search = []
        for i in range(len(active_games)):
            decide_rng = random.Random(active_seeds[i] + 77777 + len(game_states[i]))
            is_full_search.append(decide_rng.random() < full_search_fraction)

        if isinstance(current_predictor, PolicyValueModel):
            # Fast search on all active games
            run_batched_puct_search(
                games=active_games,
                model=current_predictor,
                iterations=fast_iterations,
                c_puct=c_puct,
                roots=roots_to_search,
                add_root_noise=add_root_noise,
                rng=random.Random(active_seeds[0] if active_seeds else 0),
                pass_penalty=pass_penalty,
            )
            # Additional iterations for games designated as full searches
            full_indices = [i for i, flag in enumerate(is_full_search) if flag]
            if full_indices and iterations > fast_iterations:
                full_games = [active_games[i] for i in full_indices]
                full_roots = [roots_to_search[i] for i in full_indices]
                run_batched_puct_search(
                    games=full_games,
                    model=current_predictor,
                    iterations=iterations - fast_iterations,
                    c_puct=c_puct,
                    roots=full_roots,
                    add_root_noise=False,  # noise already applied in fast search
                    rng=random.Random(active_seeds[0] if active_seeds else 0),
                    pass_penalty=pass_penalty,
                )
        else:
            for i in range(len(active_games)):
                iters = iterations if is_full_search[i] else fast_iterations
                run_puct_search(
                    game=active_games[i],
                    predictor=current_predictor,
                    iterations=iters,
                    c_puct=c_puct,
                    root=roots_to_search[i],
                    add_root_noise=add_root_noise,
                    rng=random.Random(active_seeds[i]),
                    pass_penalty=pass_penalty,
                )
        
        indices_to_remove = []
        for i in range(len(active_games)):
            game = active_games[i]
            root = roots_to_search[i]
            g_idx = active_indices[i]

            # Resignation: if current player is clearly losing, concede early
            if not game.finished and len(game_states[i]) >= resign_min_moves and root.visit_count > 0:
                if root.q < -resign_threshold:
                    game.concede()

            visit_sum = sum(ch.visit_count for ch in root.children.values())
            if visit_sum <= 0 or not root.children or game.finished or len(game_states[i]) >= max_moves:
                winner = game.score()["winner"]
                p_list = game_players[i]

                # Score-margin value targets instead of binary ±1
                if value_margin_scale > 0.0:
                    score_dict = game.score()
                    margin = score_dict["black"] - score_dict["white"]
                    scaled_margin = max(-1.0, min(1.0, margin / value_margin_scale))
                    v_list = [scaled_margin if p == BLACK else -scaled_margin for p in p_list]
                else:
                    v_list = [1.0 if p == winner else -1.0 for p in p_list]

                abs_owner = game.ownership_map()
                own_list = []
                for p in p_list:
                    mult = 1.0 if p == BLACK else -1.0
                    own_list.append(torch.tensor([mult * v for v in abs_owner], dtype=torch.float32))

                st_tensor = torch.stack(game_states[i]) if game_states[i] else torch.empty((0, INPUT_PLANES, SIZE, SIZE), dtype=torch.float32)
                pol_tensor = torch.stack(game_policies[i]) if game_policies[i] else torch.empty((0, POLICY_SIZE), dtype=torch.float32)
                val_tensor = torch.tensor(v_list, dtype=torch.float32) if v_list else torch.empty((0,), dtype=torch.float32)
                own_tensor = torch.stack(own_list) if own_list else torch.empty((0, SIZE * SIZE), dtype=torch.float32)

                res = (st_tensor, pol_tensor, val_tensor, own_tensor, len(p_list), winner, g_idx)
                if progress_queue is not None:
                    # Serialize to bytes so the queue doesn't go through torch's
                    # FD-based shared-memory reducer (which exhausts FDs on long runs).
                    buf = io.BytesIO()
                    torch.save(res, buf)
                    progress_queue.put(buf.getvalue())
                else:
                    results.append(res)
                    # In single-worker mode without a queue, print progress immediately
                    print(f"  -> Game {g_idx} finished in {len(p_list)} moves (winner: {winner})", flush=True)
                indices_to_remove.append(i)
                continue
                
            pol = torch.zeros(POLICY_SIZE, dtype=torch.float32)
            visits: dict = {}
            for move, ch in root.children.items():
                v = ch.visit_count
                visits[move] = v
                if move == PASS_MOVE:
                    pol[PASS_INDEX] = float(v) / float(visit_sum)
                else:
                    pol[move[0] * SIZE + move[1]] = float(v) / float(visit_sum)

            st = encode_game_tensor(game)[0].cpu()
            game_states[i].append(st)
            game_policies[i].append(pol)
            game_players[i].append(game.to_move)

            rng = random.Random(active_seeds[i] + len(game_states[i]))
            temp = temperature if len(game_states[i]) < 20 else max(0.05, temperature * 0.25)
            mv = _sample_from_visits(visits, rng, temp)

            if mv == PASS_MOVE:
                game.pass_turn()
                active_roots[i] = root.children.get(mv)
            elif not game.place_stone(*mv):
                game.pass_turn()
                active_roots[i] = None
            else:
                active_roots[i] = root.children.get(mv)
                
        for idx in reversed(indices_to_remove):
            active_games.pop(idx)
            active_roots.pop(idx)
            active_seeds.pop(idx)
            active_indices.pop(idx)
            game_states.pop(idx)
            game_policies.pop(idx)
            game_players.pop(idx)
            game_ownerships.pop(idx)
            start_game()
            
    return results


def _mcts_worker(start_idx, num_games, seed_base, max_moves, iterations, temperature, value_margin_scale, progress_queue):
    """Worker for parallel MCTS self-play. Writes each result to queue as it finishes."""
    for i in range(num_games):
        g_idx = start_idx + i
        seed = seed_base + 10007 * g_idx
        res = _play_one_game_mcts(
            game_idx=g_idx,
            seed=seed,
            max_moves=max_moves,
            iterations=iterations,
            temperature=temperature,
            value_margin_scale=value_margin_scale,
        )
        if progress_queue is not None:
            buf = io.BytesIO()
            torch.save(res, buf)
            progress_queue.put(buf.getvalue())


def _play_one_game_mcts(
    game_idx: int,
    seed: int,
    max_moves: int,
    iterations: int,
    temperature: float,
    value_margin_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Play a single game with MCTS and return training data."""
    import math
    rng = random.Random(seed)
    game = GoGame()
    game_states: list[torch.Tensor] = []
    game_policies: list[torch.Tensor] = []
    game_players: list[int] = []

    move_count = 0
    while not game.finished and move_count < max_moves:
        cands = candidate_moves(game)
        if not cands:
            game.pass_turn()
            move_count += 1
            continue

        root = MCTSNode(parent=None, move=None, game=game)
        mcts_search(game, root, iterations=iterations, c=math.sqrt(2), rng=rng)

        visit_sum = sum(ch.visits for ch in root.children.values())
        if visit_sum <= 0:
            game.pass_turn()
            move_count += 1
            continue

        # Record state and policy BEFORE making the move
        st = encode_game_tensor(game)[0].cpu()
        pol = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        for move, ch in root.children.items():
            pol[move[0] * SIZE + move[1]] = float(ch.visits) / float(visit_sum)
        # Pass gets 0 probability since MCTS doesn't search pass

        game_states.append(st)
        game_policies.append(pol)
        game_players.append(game.to_move)

        # Sample move by visit-count temperature
        visits = {move: ch.visits for move, ch in root.children.items()}
        temp = temperature if move_count < 20 else max(0.05, temperature * 0.25)
        mv = _sample_from_visits(visits, rng, temp)

        if not game.place_stone(*mv):
            game.pass_turn()
        move_count += 1

    winner = game.score()["winner"]
    p_list = game_players

    if value_margin_scale > 0.0:
        score_dict = game.score()
        margin = score_dict["black"] - score_dict["white"]
        scaled_margin = max(-1.0, min(1.0, margin / value_margin_scale))
        v_list = [scaled_margin if p == BLACK else -scaled_margin for p in p_list]
    else:
        v_list = [1.0 if p == winner else -1.0 for p in p_list]

    abs_owner = game.ownership_map()
    own_list = []
    for p in p_list:
        mult = 1.0 if p == BLACK else -1.0
        own_list.append(torch.tensor([mult * v for v in abs_owner], dtype=torch.float32))

    st_tensor = torch.stack(game_states) if game_states else torch.empty((0, INPUT_PLANES, SIZE, SIZE), dtype=torch.float32)
    pol_tensor = torch.stack(game_policies) if game_policies else torch.empty((0, POLICY_SIZE), dtype=torch.float32)
    val_tensor = torch.tensor(v_list, dtype=torch.float32) if v_list else torch.empty((0,), dtype=torch.float32)
    own_tensor = torch.stack(own_list) if own_list else torch.empty((0, SIZE * SIZE), dtype=torch.float32)

    return st_tensor, pol_tensor, val_tensor, own_tensor, len(p_list), winner, game_idx


def _format_eta(games_done: int, games_total: int, elapsed: float, first_completion_at: float) -> str:
    """ETA computed from the post-warmup finish rate.

    Lockstep batched MCTS finishes its first wave of ~batch_size games near-simultaneously,
    so dividing total elapsed by games_done gives a wildly pessimistic early estimate.
    Use the rate observed *after* the first completion instead.
    """
    if games_done <= 1:
        return "  warmup"
    post_warmup = elapsed - first_completion_at
    if post_warmup <= 0:
        return "  warmup"
    rate = (games_done - 1) / post_warmup  # games per second, steady-state
    remaining = games_total - games_done
    if rate <= 0:
        return "    n/a"
    return f"{remaining / rate:6.1f}s"


def main() -> None:
    require_torch()
    p = argparse.ArgumentParser(description="Generate PUCT self-play dataset")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--iterations", type=int, default=120,
                   help="MCTS iterations for 'full' searches (default: 120).")
    p.add_argument("--fast-iters", type=int, default=10,
                   help="MCTS iterations for 'fast' searches (default: 10).")
    p.add_argument("--full-search-fraction", type=float, default=0.25,
                   help="Fraction of turns that use the full iteration count (default: 0.25).")
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
    p.add_argument("--pass-penalty", type=float, default=0.0,
                   help="Tiny penalty subtracted from pass move Q during PUCT selection (default: 0.0).")
    p.add_argument("--resign-threshold", type=float, default=0.9,
                   help="Resign when root Q falls below this (default: 0.9).")
    p.add_argument("--resign-min-moves", type=int, default=30,
                   help="Minimum moves before resignation is allowed (default: 30).")
    p.add_argument("--value-margin-scale", type=float, default=20.0,
                   help="Scale factor for score-margin value targets (default: 20.0). "
                        "Set to 0.0 for binary ±1 targets.")
    p.add_argument("--engine", choices=["puct", "mcts"], default="puct",
                   help="Self-play engine: puct (neural-guided, default) or mcts "
                        "(pure Monte-Carlo tree search, no neural net needed). "
                        "Use mcts for gen 0 bootstrap when no checkpoint exists.")
    args = p.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    states, policies, values, ownerships = [], [], [], []

    def _flush(reason: str) -> None:
        blob = {
            "states": torch.cat(states) if states else torch.empty((0, INPUT_PLANES, SIZE, SIZE), dtype=torch.float32),
            "policy": torch.cat(policies) if policies else torch.empty((0, POLICY_SIZE), dtype=torch.float32),
            "value": torch.cat(values) if values else torch.empty((0,), dtype=torch.float32),
            "ownership": torch.cat(ownerships) if ownerships else torch.empty((0, SIZE * SIZE), dtype=torch.float32),
        }
        torch.save(blob, out_path)
        print(f"[{reason}] saved {blob['states'].shape[0]} samples to {out_path}", flush=True)

    started = time.perf_counter()
    games_completed = 0
    first_completion_at: Optional[float] = None  # elapsed when game 1 finishes

    if args.engine == "mcts":
        # MCTS bootstrap mode: no neural net, pure tree search.
        # MCTS is CPU-bound so we use ProcessPoolExecutor for parallelism.
        print(f"Starting MCTS bootstrap on {args.workers} worker(s)...", flush=True)
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        progress_queue = manager.Queue()

        games_per_worker = args.games // args.workers
        remainder = args.games % args.workers
        worker_tasks = []
        current_start = 0
        for i in range(args.workers):
            num = games_per_worker + (1 if i < remainder else 0)
            if num > 0:
                worker_tasks.append((
                    current_start, num, args.seed, args.max_moves,
                    args.iterations, args.temperature, args.value_margin_scale,
                    progress_queue,
                ))
                current_start += num

        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)
        futs = [pool.submit(_mcts_worker, *t) for t in worker_tasks]

        try:
            while games_completed < args.games:
                try:
                    payload = progress_queue.get(timeout=1.0)
                    res = torch.load(io.BytesIO(payload), weights_only=False)
                    st, pol, val, own, moves, winner, g_idx = res
                    states.append(st)
                    policies.append(pol)
                    values.append(val)
                    ownerships.append(own)
                    games_completed += 1
                    elapsed = time.perf_counter() - started
                    if first_completion_at is None:
                        first_completion_at = elapsed
                    eta_str = _format_eta(games_completed, args.games, elapsed, first_completion_at)
                    print(f"[{games_completed}/{args.games}] game={g_idx} samples={moves:3d} | "
                          f"winner={winner} | total={elapsed:6.1f}s | eta={eta_str}", flush=True)
                    if args.save_every > 0 and games_completed % args.save_every == 0:
                        _flush("checkpoint")
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Error reading from queue: {e}")

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
    else:
        # PUCT mode (neural-guided)
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
                fast_iterations=args.fast_iters,
                full_search_fraction=args.full_search_fraction,
                pass_penalty=args.pass_penalty,
                resign_threshold=args.resign_threshold,
                resign_min_moves=args.resign_min_moves,
                value_margin_scale=args.value_margin_scale,
            )
            for st, pol, val, own, moves, winner, g_idx in results:
                states.append(st)
                policies.append(pol)
                values.append(val)
                ownerships.append(own)
                games_completed += 1
                elapsed = time.perf_counter() - started
                if first_completion_at is None:
                    first_completion_at = elapsed
                eta_str = _format_eta(games_completed, args.games, elapsed, first_completion_at)
                print(f"[{games_completed}/{args.games}] game={g_idx} samples={moves:3d} | "
                      f"winner={winner} | total={elapsed:6.1f}s | eta={eta_str}", flush=True)
                if args.save_every > 0 and games_completed % args.save_every == 0:
                    _flush("checkpoint")
        else:
            # Multi-worker PUCT mode
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
            
            futs = [
                pool.submit(
                    _play_batch_games, *t,
                    progress_queue=progress_queue,
                    fast_iterations=args.fast_iters,
                    full_search_fraction=args.full_search_fraction,
                    pass_penalty=args.pass_penalty,
                    resign_threshold=args.resign_threshold,
                    resign_min_moves=args.resign_min_moves,
                    value_margin_scale=args.value_margin_scale,
                )
                for t in worker_tasks
            ]
            
            try:
                while games_completed < args.games:
                    try:
                        payload = progress_queue.get(timeout=1.0)
                        res = torch.load(io.BytesIO(payload), weights_only=False)
                        st, pol, val, own, moves, winner, g_idx = res
                        states.append(st)
                        policies.append(pol)
                        values.append(val)
                        ownerships.append(own)
                        games_completed += 1
                        elapsed = time.perf_counter() - started
                        if first_completion_at is None:
                            first_completion_at = elapsed
                        eta_str = _format_eta(games_completed, args.games, elapsed, first_completion_at)
                        print(f"[{games_completed}/{args.games}] game={g_idx} samples={moves:3d} | "
                              f"winner={winner} | total={elapsed:6.1f}s | eta={eta_str}", flush=True)
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
