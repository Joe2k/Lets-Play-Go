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

from ai.model import TinyPolicyValueNet, require_torch, torch
from engine.go_engine import SIZE


def _augment_batch(states: "torch.Tensor", policy: "torch.Tensor"):
    """Return (states_aug, policy_aug) with 8x samples via dihedral symmetries.

    states: [B, C, H, W]; policy: [B, H*W].
    """
    B, C, H, W = states.shape
    assert H == W == SIZE
    pol_2d = policy.view(B, H, W)

    aug_s = []
    aug_p = []
    for k in range(4):
        s_rot = torch.rot90(states, k=k, dims=(2, 3))
        p_rot = torch.rot90(pol_2d, k=k, dims=(1, 2))
        aug_s.append(s_rot)
        aug_p.append(p_rot)
        # Horizontal flip on top of rotation = the 4 reflective members.
        aug_s.append(torch.flip(s_rot, dims=(3,)))
        aug_p.append(torch.flip(p_rot, dims=(2,)))

    s_out = torch.cat(aug_s, dim=0)
    p_out = torch.cat(aug_p, dim=0).reshape(8 * B, H * W)
    return s_out, p_out


def main() -> None:
    require_torch()
    p = argparse.ArgumentParser(description="Train policy-value model")
    p.add_argument("--data", type=str, default="data/selfplay.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
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

    blob = torch.load(args.data, map_location="cpu")
    states = blob["states"].float()
    policy = blob["policy"].float()
    value = blob["value"].float()
    n = states.shape[0]
    if n == 0:
        raise RuntimeError("dataset is empty; run generate_selfplay_data.py first")

    device = torch.device(args.device)
    model = TinyPolicyValueNet().to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"initialized weights from {args.init_from}")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(reason: str) -> None:
        torch.save({"state_dict": model.state_dict()}, out_path)
        print(f"[{reason}] saved checkpoint to {out_path}")

    indices = list(range(n))
    try:
        for epoch in range(1, args.epochs + 1):
            random.shuffle(indices)
            epoch_loss = 0.0
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
                policy_loss = -(p_t * logp).sum(dim=1).mean()
                value_loss = torch.mean((v_pred - v_t) ** 2)
                loss = policy_loss + value_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item())
                batches += 1

            print(f"epoch {epoch}/{args.epochs} loss={epoch_loss / max(1, batches):.5f}")
            _save(f"epoch {epoch}")
    except KeyboardInterrupt:
        print("\nInterrupted — saving latest weights.")
        _save("interrupted")
        sys.exit(0)

    _save("done")


if __name__ == "__main__":
    main()
