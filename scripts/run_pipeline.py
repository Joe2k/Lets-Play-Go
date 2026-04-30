"""Iterative self-play / train / gate pipeline.

Runs generations 0..N end-to-end:

    gen 0:  uniform predictor -> data_g0 -> train pv_g0 (best := pv_g0)
    gen i:  best -> data_gi -> train pv_gi (init-from best) ->
            gate eval pv_gi vs best -> automatically promote to save time

State (current best checkpoint, per-gen status) is persisted in
`<run-dir>/state.json` so re-running the script resumes where it left
off — completed steps are skipped automatically.

Each generation also appends a row to `<run-dir>/pipeline.log.jsonl`
with timing, win rate, and promotion outcome.

Stop the script at any time with Ctrl+C; the underlying scripts each
have their own checkpointing so accumulated work is preserved.

Example:

    .venv/bin/python scripts/run_pipeline.py --generations 4 \\
        --games 30 --self-play-iters 200 \\
        --epochs 10 --eval-games 20 --eval-iters 200 \\
        --gate-threshold 0.52 --device mps

To resume the same run later:

    .venv/bin/python scripts/run_pipeline.py --generations 6 \\
        --run-dir runs/default
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _run(cmd: list[str]) -> int:
    """Run a subprocess, streaming stdout/stderr to the parent terminal."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd)


def _check(cmd: list[str], allow_codes: tuple[int, ...] = (0,)) -> int:
    rc = _run(cmd)
    if rc not in allow_codes:
        print(f"command exited with code {rc}; aborting pipeline.", file=sys.stderr)
        sys.exit(rc)
    return rc


class PipelineState:
    def __init__(self, run_dir: pathlib.Path) -> None:
        self.run_dir = run_dir
        self.state_path = run_dir / "state.json"
        self.log_path = run_dir / "pipeline.log.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            with self.state_path.open("r") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "best_checkpoint": None,
                "best_gen": None,
                "generations": {},  # str(gen_idx) -> dict
            }
            self._save()

    def _save(self) -> None:
        with self.state_path.open("w") as f:
            json.dump(self.data, f, indent=2)

    def gen(self, idx: int) -> dict:
        return self.data["generations"].setdefault(str(idx), {})

    def update_gen(self, idx: int, **kwargs) -> None:
        self.gen(idx).update(kwargs)
        self._save()

    def set_best(self, gen_idx: int, checkpoint: str) -> None:
        self.data["best_checkpoint"] = checkpoint
        self.data["best_gen"] = gen_idx
        self._save()

    def best_checkpoint(self) -> Optional[str]:
        return self.data.get("best_checkpoint")

    def append_log(self, entry: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def _self_play(
    state: PipelineState,
    gen_idx: int,
    out_path: pathlib.Path,
    games: int,
    iterations: int,
    device: str,
    seed_base: int,
    workers: int = 1,
    batch_size: int = 32,
    c_puct: float = 1.25,
    fast_iters: int = 10,
    full_search_fraction: float = 0.25,
    pass_penalty: float = 0.0,
    resign_threshold: float = 0.9,
    resign_min_moves: int = 30,
    value_margin_scale: float = 20.0,
) -> None:
    if out_path.exists() and state.gen(gen_idx).get("self_play_done"):
        print(f"[gen {gen_idx}] self-play already complete, skipping.")
        return

    # Check for partial data to resume from
    start_game = 0
    if out_path.exists():
        try:
            import torch as _torch
            existing = _torch.load(out_path, weights_only=False)
            start_game = existing.get("games_completed", 0)
            if start_game >= games:
                print(f"[gen {gen_idx}] self-play already complete ({start_game}/{games} games), skipping.")
                state.update_gen(gen_idx, self_play_done=True)
                return
            elif start_game > 0:
                print(f"[gen {gen_idx}] resuming from game {start_game}/{games} (existing file: {out_path})")
        except Exception:
            print(f"[gen {gen_idx}] warning: could not read existing file, starting fresh")
            start_game = 0

    predictor = state.best_checkpoint()
    is_bootstrap = predictor is None

    remaining = games - start_game
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "generate_selfplay_data.py"),
        "--games", str(remaining),
        "--start-game", str(start_game),
        "--iterations", str(iterations),
        "--fast-iters", str(fast_iters),
        "--full-search-fraction", str(full_search_fraction),
        "--pass-penalty", str(pass_penalty),
        "--resign-threshold", str(resign_threshold),
        "--resign-min-moves", str(resign_min_moves),
        "--value-margin-scale", str(value_margin_scale),
        "--seed", str(seed_base + 100 * gen_idx),
        "--output", str(out_path),
        "--device", device,
        "--workers", str(workers),
        "--batch-size", str(batch_size),
    ]

    if is_bootstrap:
        # Gen 0: use pure MCTS (no neural net) for strong bootstrap data.
        # MCTS doesn't need c_puct or a model checkpoint.
        cmd += ["--engine", "mcts"]
        print(f"[gen {gen_idx}] Bootstrap: using MCTS engine (no neural net)")
    else:
        cmd += [
            "--engine", "puct",
            "--c-puct", str(c_puct),
            "--predictor-checkpoint", predictor,
        ]

    started = time.perf_counter()
    _check(cmd)
    elapsed = time.perf_counter() - started
    state.update_gen(gen_idx, self_play_done=True, self_play_seconds=elapsed,
                     self_play_path=str(out_path))


def _train(
    state: PipelineState,
    gen_idx: int,
    data_paths: list[pathlib.Path],
    out_path: pathlib.Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    ownership_weight: float = 0.3,
) -> None:
    if out_path.exists() and state.gen(gen_idx).get("train_done"):
        print(f"[gen {gen_idx}] training already complete, skipping.")
        return

    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "train_policy_value.py"),
        "--data", *[str(p) for p in data_paths],
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--ownership-weight", str(ownership_weight),
        "--out", str(out_path),
        "--device", device,
    ]
    init_from = state.best_checkpoint()
    if init_from is not None:
        cmd += ["--init-from", init_from]

    started = time.perf_counter()
    _check(cmd)
    elapsed = time.perf_counter() - started
    state.update_gen(gen_idx, train_done=True, train_seconds=elapsed,
                     checkpoint_path=str(out_path))


def _gate_eval(
    state: PipelineState,
    gen_idx: int,
    new_checkpoint: pathlib.Path,
    games: int,
    iterations: int,
    threshold: float,
    csv_path: pathlib.Path,
    c_puct: float = 1.25,
    workers: int = 1,
    eval_temperature: float = 0.25,
) -> tuple[bool, Optional[float]]:
    """Returns (promoted, win_rate). Win rate is None if eval was skipped."""
    if state.gen(gen_idx).get("eval_done"):
        wr = state.gen(gen_idx).get("eval_winrate")
        promoted = state.gen(gen_idx).get("promoted", False)
        print(f"[gen {gen_idx}] gate eval already complete (winrate={wr}, promoted={promoted}); skipping.")
        return promoted, wr

    best = state.best_checkpoint()
    if best is None:
        # Gen 0: nothing to compare against — auto-promote.
        state.update_gen(gen_idx, eval_done=True, eval_winrate=None, promoted=True)
        return True, None

    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "eval_match.py"),
        "--games", str(games),
        "--a-name", f"gen{gen_idx}",
        "--a-engine", "puct",
        "--a-model", str(new_checkpoint),
        "--a-iters", str(iterations),
        "--b-name", f"best_gen{state.data.get('best_gen')}",
        "--b-engine", "puct",
        "--b-model", best,
        "--b-iters", str(iterations),
        "--csv", str(csv_path),
        "--gate-threshold", str(threshold),
        "--c-puct", str(c_puct),
        "--workers", str(workers),
        "--eval-temperature", str(eval_temperature),
    ]
    started = time.perf_counter()
    _run(cmd)
    elapsed = time.perf_counter() - started
    win_rate = _winrate_from_csv(csv_path, expected_a_name=f"gen{gen_idx}")
    promoted = win_rate is not None and win_rate >= threshold
    state.update_gen(
        gen_idx,
        eval_done=True,
        eval_seconds=elapsed,
        eval_winrate=win_rate,
        promoted=promoted,
    )
    return promoted, win_rate


def _winrate_from_csv(csv_path: pathlib.Path, expected_a_name: str) -> Optional[float]:
    """Compute A's win rate from the most recent match_id matching expected_a_name."""
    if not csv_path.exists():
        return None
    import csv as _csv
    with csv_path.open("r", newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return None
    # Find the latest match_id that involves expected_a_name.
    relevant = [r for r in rows if expected_a_name in (r.get("black_name"), r.get("white_name"))]
    if not relevant:
        return None
    last_match_id = relevant[-1]["match_id"]
    match_rows = [r for r in relevant if r["match_id"] == last_match_id]
    if not match_rows:
        return None
    wins = sum(1 for r in match_rows if r["winner_name"] == expected_a_name)
    return wins / len(match_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Run iterative RL pipeline")
    p.add_argument("--generations", type=int, default=3,
                   help="Number of generations to run (gen_0 through gen_{N-1}).")
    p.add_argument("--start-gen", type=int, default=0,
                   help="Resume from this generation (skip earlier gens).")
    p.add_argument("--run-dir", type=str, default="runs/default",
                   help="Directory holding state.json, datasets, checkpoints, and logs.")
    p.add_argument("--games", type=int, default=60,
                   help="Self-play games per generation.")
    p.add_argument("--self-play-iters", type=int, default=400,
                   help="MCTS/PUCT iterations per move during self-play.")
    p.add_argument("--self-play-fast-iters", type=int, default=200,
                   help="MCTS/PUCT iterations for fast searches during self-play.")
    p.add_argument("--self-play-full-search-fraction", type=float, default=0.5,
                   help="Fraction of self-play turns that use the full iteration count.")
    p.add_argument("--self-play-pass-penalty", type=float, default=0.0,
                   help="Tiny penalty subtracted from pass move Q during self-play PUCT search.")
    p.add_argument("--self-play-resign-threshold", type=float, default=0.95,
                   help="Resign when root Q falls below this during self-play.")
    p.add_argument("--self-play-resign-min-moves", type=int, default=30,
                   help="Minimum moves before resignation is allowed in self-play.")
    p.add_argument("--self-play-value-margin-scale", type=float, default=15.0,
                   help="Scale factor for score-margin value targets (0 = binary ±1). "
                        "15 is appropriate for 9x9; 20 compresses typical margins too much.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate. When fine-tuning from a prior checkpoint, "
                        "consider 3e-4 to avoid destabilizing the network.")
    p.add_argument("--ownership-weight", type=float, default=0.3,
                   help="Weight for ownership MSE loss term (default: 0.3). "
                        "Lower this when the model is weak so policy/value get priority.")
    p.add_argument("--eval-games", type=int, default=40,
                   help="Games per gate-eval match. Below ~30 the win-rate "
                        "estimate is too noisy to gate at 0.55.")
    p.add_argument("--eval-iters", type=int, default=400)
    p.add_argument("--eval-temperature", type=float, default=0.25,
                   help="Temperature for visit-count sampling during gate eval. "
                        "0 = deterministic argmax. Small values (0.25) create "
                        "game diversity without corrupting search with Dirichlet noise.")
    p.add_argument("--gate-threshold", type=float, default=0.55,
                   help="Win rate needed to promote a new checkpoint.")
    p.add_argument("--replay-window", type=int, default=5,
                   help="Train each generation on the last N self-play datasets "
                        "(sliding replay buffer). 1 = each gen sees only its own data.")
    p.add_argument("--c-puct", type=float, default=1.25,
                   help="PUCT exploration constant for gate eval. "
                        "Lower = more exploitation for accurate win-rate estimation.")
    p.add_argument("--self-play-c-puct", type=float, default=1.5,
                   help="PUCT exploration constant for self-play. "
                        "Higher than eval (1.5 vs 1.25) because the model is weaker "
                        "during search and needs more exploration to discover good moves.")
    p.add_argument("--workers", type=int, default=1,
                   help="Default worker count (used by both self-play and eval "
                        "unless overridden by --self-play-workers / --eval-workers).")
    p.add_argument("--self-play-workers", type=int, default=None,
                   help="Workers for self-play data generation. Defaults to --workers. "
                        "On a single GPU prefer few workers (1-2) with a large batch; "
                        "GPU utilization comes from --self-play-batch-size, not workers.")
    p.add_argument("--eval-workers", type=int, default=None,
                   help="Workers for gate eval. Defaults to --workers. Eval games run "
                        "independently with no cross-batching, so more workers ~= more "
                        "throughput up to your CPU/GPU limit.")
    p.add_argument("--self-play-batch-size", type=int, default=None,
                   help="Concurrent games per self-play worker (lockstep batched MCTS). "
                        "Defaults to --batch-size. Larger = more GPU saturation per worker.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed-base", type=int, default=0)
    args = p.parse_args()
    if args.replay_window < 1:
        raise SystemExit("--replay-window must be >= 1")

    self_play_workers = args.self_play_workers if args.self_play_workers is not None else args.workers
    eval_workers = args.eval_workers if args.eval_workers is not None else args.workers
    self_play_batch = args.self_play_batch_size if args.self_play_batch_size is not None else args.batch_size

    run_dir = pathlib.Path(args.run_dir)
    data_dir = run_dir / "data"
    ckpt_dir = run_dir / "checkpoints"
    eval_csv = run_dir / "eval.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = PipelineState(run_dir)
    print(f"=== pipeline start @ {_now()} ===")
    print(f"run_dir: {run_dir}")
    print(f"current best: {state.best_checkpoint() or '(none)'}\n")

    overall_started = time.perf_counter()
    for gen in range(args.start_gen, args.generations):
        gen_started = time.perf_counter()
        print(f"\n========== generation {gen} ==========")
        data_path = data_dir / f"selfplay_g{gen}.pt"
        ckpt_path = ckpt_dir / f"pv_g{gen}.pt"

        _self_play(state, gen, data_path,
                   games=args.games,
                   iterations=args.self_play_iters,
                   device=args.device,
                   seed_base=args.seed_base,
                   workers=self_play_workers,
                   batch_size=self_play_batch,
                   c_puct=args.self_play_c_puct,
                   fast_iters=args.self_play_fast_iters,
                   full_search_fraction=args.self_play_full_search_fraction,
                   pass_penalty=args.self_play_pass_penalty,
                   resign_threshold=args.self_play_resign_threshold,
                   resign_min_moves=args.self_play_resign_min_moves,
                   value_margin_scale=args.self_play_value_margin_scale)

        # Sliding replay window: train on the last `replay_window`
        # self-play datasets that actually exist on disk.
        window_start = max(0, gen - args.replay_window + 1)
        train_paths: list[pathlib.Path] = []
        for g in range(window_start, gen + 1):
            p_path = data_dir / f"selfplay_g{g}.pt"
            if p_path.exists():
                train_paths.append(p_path)
        print(f"[gen {gen}] replay window: {[p.name for p in train_paths]}")

        _train(state, gen, train_paths, ckpt_path,
               epochs=args.epochs,
               batch_size=args.batch_size,
               lr=args.lr,
               device=args.device,
               ownership_weight=args.ownership_weight)

        promoted, win_rate = _gate_eval(
            state, gen, ckpt_path,
            games=args.eval_games,
            iterations=args.eval_iters,
            threshold=args.gate_threshold,
            csv_path=eval_csv,
            c_puct=args.c_puct,
            workers=eval_workers,
            eval_temperature=args.eval_temperature,
        )

        if promoted:
            state.set_best(gen, str(ckpt_path))
            verdict = "PROMOTED"
        else:
            verdict = "REJECTED (keeping previous best)"
        gen_elapsed = time.perf_counter() - gen_started
        wr_str = f"{win_rate*100:.1f}%" if win_rate is not None else "n/a"

        log_entry = {
            "timestamp": _now(),
            "gen": gen,
            "win_rate": win_rate,
            "promoted": promoted,
            "best_after": state.best_checkpoint(),
            "elapsed_seconds": gen_elapsed,
        }
        state.append_log(log_entry)

        print(f"\n[gen {gen}] {verdict} | win_rate={wr_str} | gen_time={gen_elapsed:.1f}s")
        print(f"[gen {gen}] best_checkpoint={state.best_checkpoint()}")

    print(f"\n=== pipeline finished | total {time.perf_counter()-overall_started:.1f}s ===")
    print(f"final best: {state.best_checkpoint()}")
    print(f"per-gen log: {state.log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted. State saved — re-run the same command to resume.")
        sys.exit(0)
