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
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.model import PolicyValueModel, encode_game_tensor, require_torch, torch
from ai.puct_agent import run_puct_search
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
    args = p.parse_args()

    rng = random.Random(args.seed)
    if args.predictor_checkpoint:
        predictor = PolicyValueModel(model_path=args.predictor_checkpoint, device=args.device)
        print(f"using predictor checkpoint: {args.predictor_checkpoint}")
    else:
        predictor = _UniformPredictor()
        print("using uniform bootstrap predictor")

    add_root_noise = not args.no_noise

    states = []
    policies = []
    player_to_move = []
    values = []

    for gidx in range(args.games):
        game = GoGame()
        samples_idx = []
        for ply in range(args.max_moves):
            if game.finished:
                break
            root = run_puct_search(
                game=game,
                predictor=predictor,
                iterations=args.iterations,
                c_puct=1.4,
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
            states.append(st)
            policies.append(pol)
            player_to_move.append(game.to_move)
            samples_idx.append(len(states) - 1)

            temp = args.temperature if ply < 20 else max(0.05, args.temperature * 0.25)
            mv = _sample_from_visits(visits, rng=rng, temperature=temp)
            if not game.place_stone(*mv):
                # Defensive fallback.
                game.pass_turn()
                break

        winner = game.score()["winner"]
        for idx in samples_idx:
            values.append(1.0 if player_to_move[idx] == winner else -1.0)
        print(f"game {gidx+1}/{args.games}: samples={len(samples_idx)} winner={winner}")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "states": torch.stack(states) if states else torch.empty((0, 3, 9, 9), dtype=torch.float32),
        "policy": torch.stack(policies) if policies else torch.empty((0, 81), dtype=torch.float32),
        "value": torch.tensor(values, dtype=torch.float32) if values else torch.empty((0,), dtype=torch.float32),
    }
    torch.save(blob, out_path)
    print(f"saved {blob['states'].shape[0]} samples to {out_path}")


if __name__ == "__main__":
    main()
