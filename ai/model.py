"""Tiny policy-value CNN for 9x9 Go.

This is an inference-first scaffold:
- policy head predicts logits for all 81 points
- value head predicts win value in [-1, 1] from side-to-move perspective
"""

from __future__ import annotations

from typing import Optional

from engine.go_engine import BLACK, SIZE, WHITE, GoGame

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None


def torch_available() -> bool:
    return torch is not None and nn is not None


def require_torch() -> None:
    if not torch_available():
        raise RuntimeError(
            "PyTorch is required for --engine puct. Install it with "
            "`python3 -m pip install torch`."
        )


if torch_available():
    class TinyPolicyValueNet(nn.Module):  # type: ignore[misc]
        def __init__(self, channels: int = 64) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels, 2, kernel_size=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * SIZE * SIZE, SIZE * SIZE),
            )
            self.value_head = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(SIZE * SIZE, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            h = self.trunk(x)
            policy_logits = self.policy_head(h)
            value = self.value_head(h).squeeze(-1)
            return policy_logits, value
else:
    class TinyPolicyValueNet:  # pragma: no cover - unavailable dependency path
        def __init__(self, channels: int = 64) -> None:
            raise RuntimeError("PyTorch is required for TinyPolicyValueNet.")


def encode_game_tensor(game: GoGame):
    """Return [1, 3, 9, 9] float tensor from side-to-move perspective."""
    require_torch()
    me = game.to_move
    opp = WHITE if me == BLACK else BLACK
    p0 = []
    p1 = []
    p2 = []
    for r in range(SIZE):
        row_me = []
        row_opp = []
        row_turn = []
        for c in range(SIZE):
            v = game.board[r * SIZE + c]
            row_me.append(1.0 if v == me else 0.0)
            row_opp.append(1.0 if v == opp else 0.0)
            row_turn.append(1.0)
        p0.append(row_me)
        p1.append(row_opp)
        p2.append(row_turn)
    x = torch.tensor([p0, p1, p2], dtype=torch.float32).unsqueeze(0)
    return x


class PolicyValueModel:
    """Thin wrapper for inference and optional checkpoint loading."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu") -> None:
        require_torch()
        self.device = torch.device(device)
        self.net = TinyPolicyValueNet().to(self.device)
        self.net.eval()
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.net.load_state_dict(state)

    def predict(self, game: GoGame) -> tuple[list[float], float]:
        with torch.no_grad():
            x = encode_game_tensor(game).to(self.device)
            logits, value = self.net(x)
            probs = torch.softmax(logits[0], dim=0).cpu().tolist()
            v = float(value[0].item())
        return probs, v
