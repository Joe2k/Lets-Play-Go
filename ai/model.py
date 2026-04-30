"""Residual policy-value CNN for 9x9 Go.

Implemented as a small ResNet to provide enough capacity for strategic learning
while remaining fast enough for CPU-based MCTS.
"""

from __future__ import annotations

from typing import Optional

from engine.go_engine import BLACK, SIZE, WHITE, GoGame

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


# 8-plane input, AlphaZero-style:
#   0  own stones at t
#   1  opponent stones at t
#   2  own stones at t-1
#   3  opponent stones at t-1
#   4  own stones at t-2
#   5  opponent stones at t-2
#   6  color-to-move (1.0 if Black to move else 0.0; broadcast to full plane)
#   7  legal-moves mask (1.0 where the candidate filter would let us play)
INPUT_PLANES = 8

# Policy: 81 board moves + 1 pass slot.
POLICY_SIZE = SIZE * SIZE + 1
PASS_INDEX = SIZE * SIZE

# Ownership: one scalar per board point.
OWNERSHIP_SIZE = SIZE * SIZE


def torch_available() -> bool:
    return torch is not None and nn is not None


def require_torch() -> None:
    if not torch_available():
        raise RuntimeError(
            "PyTorch is required for --engine puct. Install it with "
            "`python3 -m pip install torch`."
        )


if torch_available():
    class ResBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return torch.relu(out)

    class TinyPolicyValueNet(nn.Module):
        def __init__(self, channels: int = 96, num_blocks: int = 8) -> None:
            super().__init__()
            self.start_conv = nn.Sequential(
                nn.Conv2d(INPUT_PLANES, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            )

            self.res_tower = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

            self.policy_head = nn.Sequential(
                nn.Conv2d(channels, 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * SIZE * SIZE, POLICY_SIZE),
            )

            self.value_head = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(SIZE * SIZE, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )

            self.ownership_head = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.start_conv(x)
            h = self.res_tower(h)
            policy_logits = self.policy_head(h)
            value = self.value_head(h).squeeze(-1)
            ownership = self.ownership_head(h)
            return policy_logits, value, ownership
else:
    class TinyPolicyValueNet:  # pragma: no cover
        def __init__(self, channels: int = 96, num_blocks: int = 8) -> None:
            raise RuntimeError("PyTorch is required for TinyPolicyValueNet.")


def _legal_mask(game: GoGame) -> list[float]:
    """1.0 wherever the candidate filter would let the to-move player play."""
    # Local import to avoid a circular dependency (ai.mcts imports go_engine).
    from .mcts import candidate_moves
    mask = [0.0] * (SIZE * SIZE)
    for r, c in candidate_moves(game):
        mask[r * SIZE + c] = 1.0
    return mask


def _planes_for_game(game: GoGame) -> list[list[float]]:
    """Return 8 flat planes (each length 81) for `game`."""
    me = game.to_move
    opp = WHITE if me == BLACK else BLACK
    board = game.board
    n = SIZE * SIZE

    own_t = [1.0 if board[i] == me else 0.0 for i in range(n)]
    opp_t = [1.0 if board[i] == opp else 0.0 for i in range(n)]

    if len(game.prev_boards) >= 1:
        b1 = game.prev_boards[0]
        own_t1 = [1.0 if b1[i] == me else 0.0 for i in range(n)]
        opp_t1 = [1.0 if b1[i] == opp else 0.0 for i in range(n)]
    else:
        own_t1 = [0.0] * n
        opp_t1 = [0.0] * n

    if len(game.prev_boards) >= 2:
        b2 = game.prev_boards[1]
        own_t2 = [1.0 if b2[i] == me else 0.0 for i in range(n)]
        opp_t2 = [1.0 if b2[i] == opp else 0.0 for i in range(n)]
    else:
        own_t2 = [0.0] * n
        opp_t2 = [0.0] * n

    color_plane = [1.0 if me == BLACK else 0.0] * n
    legal = _legal_mask(game)
    return [own_t, opp_t, own_t1, opp_t1, own_t2, opp_t2, color_plane, legal]


def encode_game_tensor(game: GoGame) -> torch.Tensor:
    """Return [1, INPUT_PLANES, 9, 9] float tensor from side-to-move perspective."""
    require_torch()
    planes = _planes_for_game(game)
    t = torch.tensor(planes, dtype=torch.float32).view(INPUT_PLANES, SIZE, SIZE)
    return t.unsqueeze(0)


def encode_games_batch(games: list[GoGame], device: Optional[torch.device] = None) -> torch.Tensor:
    """Return [N, INPUT_PLANES, 9, 9] float tensor for a batch of games."""
    require_torch()
    if not games:
        return torch.empty((0, INPUT_PLANES, SIZE, SIZE), dtype=torch.float32, device=device)
    # Per-game plane construction in Python is fine — the 9x9 board makes the
    # constant factor negligible, and avoiding the previous all-tensor path
    # keeps history / legal-mask logic readable.
    stacks = [_planes_for_game(g) for g in games]
    t = torch.tensor(stacks, dtype=torch.float32, device=device)
    return t.view(len(games), INPUT_PLANES, SIZE, SIZE)


class PolicyValueModel:
    """Thin wrapper for inference and checkpoint loading."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu") -> None:
        require_torch()
        self.device = torch.device(device)
        self.net = TinyPolicyValueNet().to(self.device)
        self.net.eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            if missing:
                print(f"Warning: missing keys when loading checkpoint: {missing}")
            if unexpected:
                print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")

    def predict(self, game: GoGame) -> tuple[list[float], float]:
        probs_list, values = self.predict_batch([game])
        return probs_list[0], values[0]

    def predict_batch(self, games: list[GoGame]) -> tuple[list[list[float]], list[float]]:
        if not games:
            return [], []
        with torch.no_grad():
            x = encode_games_batch(games, device=self.device)
            logits, values, _ownership = self.net(x)
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            vs = values.cpu().tolist()
        return probs, vs
