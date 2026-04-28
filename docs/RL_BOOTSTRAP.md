# CNN + RL Bootstrap

Pipeline for training a policy-value network through iterative self-play. Each generation: self-play with the current best net → train new net → gate-eval → promote if it wins.

## 1) Install PyTorch

```bash
.venv/bin/pip install torch
```

## 2) Generate self-play dataset

Bootstrap with a uniform predictor (no model yet):

```bash
.venv/bin/python scripts/generate_selfplay_data.py \
    --games 30 --iterations 200 \
    --output data/selfplay_g0.pt
```

Or generate with a trained checkpoint as the predictor:

```bash
.venv/bin/python scripts/generate_selfplay_data.py \
    --games 30 --iterations 200 \
    --predictor-checkpoint checkpoints/pv_g0.pt \
    --output data/selfplay_g1.pt
```

Dirichlet root noise is on by default during self-play (turn it off only for debugging with `--no-noise`).

Output tensors:
- `states`: `[N, 3, 9, 9]`
- `policy`: `[N, 81]` (visit-count distribution from MCTS)
- `value`: `[N]` in {-1, +1}

## 3) Train a network

Initial generation:

```bash
.venv/bin/python scripts/train_policy_value.py \
    --data data/selfplay_g0.pt --epochs 10 \
    --out checkpoints/pv_g0.pt
```

Subsequent generations — fine-tune from the previous checkpoint:

```bash
.venv/bin/python scripts/train_policy_value.py \
    --data data/selfplay_g1.pt --epochs 10 \
    --init-from checkpoints/pv_g0.pt \
    --out checkpoints/pv_g1.pt
```

Training augments each sample with the 8 dihedral symmetries by default. Disable with `--no-augment` if you need a raw-data ablation.

## 4) Gate-eval the new net before promoting

Only promote `pv_g1` if it beats `pv_g0` by at least 55%:

```bash
.venv/bin/python scripts/eval_match.py --games 40 \
    --a-name new --a-engine puct --a-model checkpoints/pv_g1.pt --a-iters 200 \
    --b-name old --b-engine puct --b-model checkpoints/pv_g0.pt --b-iters 200 \
    --gate-threshold 0.55
```

Exit code is `0` on pass, `1` on fail. Useful for scripting the loop.

## 5) Play the trained model

```bash
.venv/bin/python play.py --engine puct --ai-iterations 300 \
    --model-path checkpoints/pv_g1.pt
```

## Iterative loop, end-to-end

```
gen_0:  uniform → self-play g0 → train pv_g0
gen_1:  pv_g0   → self-play g1 → train pv_g1 (init-from pv_g0) → gate vs pv_g0
gen_2:  pv_g1   → self-play g2 → train pv_g2 (init-from pv_g1) → gate vs pv_g1
...
```

If a generation fails the gate, do one of:
- collect more games at the same generation,
- bump iterations during self-play,
- lower the temperature schedule.

## Reference

| Flag | Where | Default | Purpose |
|---|---|---|---|
| `--predictor-checkpoint` | generate_selfplay_data | none (uniform) | drives self-play with trained net |
| `--no-noise` | generate_selfplay_data | off | disables Dirichlet root noise |
| `--init-from` | train_policy_value | none | warm-start from previous checkpoint |
| `--no-augment` | train_policy_value | off | disables 8-fold symmetry augmentation |
| `--gate-threshold` | eval_match | none | exit non-zero if A win rate < threshold |
| `--a-engine` / `--b-engine` | eval_match | mcts | choose `puct` or `mcts` per side |
| `--a-model` / `--b-model` | eval_match | none | checkpoint paths for PUCT sides |
