"""Train TinyPolicyValueNet from self-play dataset.

Augments each (state, policy, value) sample with the 8 board symmetries
(4 rotations x {identity, horizontal flip}) — Go is invariant under these,
so this is essentially free 8x training data.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.model import POLICY_SIZE, TinyPolicyValueNet, require_torch, torch
from engine.go_engine import SIZE


def _augment_batch(states: "torch.Tensor", policy: "torch.Tensor"):
    """Return (states_aug, policy_aug) with 8x samples via dihedral symmetries.

    states: [B, C, H, W]; policy: [B, POLICY_SIZE] = [B, H*W + 1] (last col = pass).
    Pass is invariant under board symmetries; only the H*W board slice rotates.
    """
    B, C, H, W = states.shape
    assert H == W == SIZE
    assert policy.shape[1] == POLICY_SIZE
    pol_board = policy[:, :H * W].view(B, H, W)
    pol_pass = policy[:, H * W:]  # [B, 1]

    aug_s = []
    aug_p = []
    for k in range(4):
        s_rot = torch.rot90(states, k=k, dims=(2, 3))
        p_rot = torch.rot90(pol_board, k=k, dims=(1, 2))
        aug_s.append(s_rot)
        aug_p.append(torch.cat([p_rot.reshape(B, H * W), pol_pass], dim=1))
        # Horizontal flip on top of rotation = the 4 reflective members.
        aug_s.append(torch.flip(s_rot, dims=(3,)))
        aug_p.append(
            torch.cat([torch.flip(p_rot, dims=(2,)).reshape(B, H * W), pol_pass], dim=1)
        )

    s_out = torch.cat(aug_s, dim=0)
    p_out = torch.cat(aug_p, dim=0)
    return s_out, p_out


def main() -> None:
    require_torch()
    p = argparse.ArgumentParser(description="Train policy-value model")
    p.add_argument("--data", type=str, nargs="+", default=["data/selfplay.pt"],
                   help="One or more self-play datasets to concatenate (replay window).")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-5,
                   help="Minimum learning rate for decay scheduler.")
    p.add_argument("--patience", type=int, default=3,
                   help="Number of epochs to wait for improvement before early stopping.")
    p.add_argument("--min-delta", type=float, default=0.01,
                   help="Minimum change in loss to qualify as an improvement.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/pv_latest.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--init-from", type=str, default=None,
                   help="Optional checkpoint to fine-tune from.")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable 8-fold symmetry augmentation.")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    states_chunks: list["torch.Tensor"] = []
    policy_chunks: list["torch.Tensor"] = []
    value_chunks: list["torch.Tensor"] = []
    for path in args.data:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        states_chunks.append(blob["states"].float())
        policy_chunks.append(blob["policy"].float())
        value_chunks.append(blob["value"].float())
        print(f"loaded {blob['states'].shape[0]} samples from {path}", flush=True)
    states = torch.cat(states_chunks, dim=0)
    policy = torch.cat(policy_chunks, dim=0)
    value = torch.cat(value_chunks, dim=0)
    n = states.shape[0]
    if n == 0:
        raise RuntimeError("dataset is empty; run generate_selfplay_data.py first")
    print(f"training on {n} total samples from {len(args.data)} file(s)", flush=True)

    device = torch.device(args.device)
    model = TinyPolicyValueNet().to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"initialized weights from {args.init_from}")
    model.train()
    # Use AdamW with weight decay to prevent overfitting on small self-play datasets.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr_min
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(reason: str) -> None:
        torch.save({"state_dict": model.state_dict()}, out_path)
        print(f"[{reason}] saved checkpoint to {out_path}", flush=True)

    indices = list(range(n))
    best_loss = float("inf")
    epochs_without_improvement = 0
    
    try:
        for epoch in range(1, args.epochs + 1):
            random.shuffle(indices)
            epoch_loss = 0.0
            epoch_p_loss = 0.0
            epoch_v_loss = 0.0
            batches = 0
            for s in range(0, n, args.batch_size):
                batch_idx = indices[s:s + args.batch_size]
                x = states[batch_idx].to(device)
                p_t = policy[batch_idx].to(device)
                v_t = value[batch_idx].to(device)

                if not args.no_augment:
                    x, p_t = _augment_batch(x, p_t)
                    v_t = v_t.repeat(8)

                logits, v_pred = model(x)
                logp = torch.log_softmax(logits, dim=1)
                
                # Cross-entropy for policy
                policy_loss = -(p_t * logp).sum(dim=1).mean()
                # Mean Squared Error for value
                value_loss = torch.mean((v_pred - v_t) ** 2)
                loss = policy_loss + value_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item())
                epoch_p_loss += float(policy_loss.item())
                epoch_v_loss += float(value_loss.item())
                batches += 1

            avg_loss = epoch_loss / batches
            print(
                f"epoch {epoch:2d}/{args.epochs} | "
                f"loss={avg_loss:.4f} | "
                f"p_loss={epoch_p_loss / batches:.4f} | "
                f"v_loss={epoch_v_loss / batches:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True
            )
            
            scheduler.step()
            _save(f"epoch {epoch}")

            # Early stopping check
            if avg_loss < (best_loss - args.min_delta):
                best_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping triggered: loss has not improved for {args.patience} epochs.")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted — saving latest weights.")
        _save("interrupted")
        sys.exit(0)

    _save("done")


if __name__ == "__main__":
    main()
