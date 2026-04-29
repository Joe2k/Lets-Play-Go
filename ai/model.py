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
        def __init__(self, channels: int = 64, num_blocks: int = 6) -> None:
            super().__init__()
            # Initial convolution
            self.start_conv = nn.Sequential(
                nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            
            # Residual tower
            self.res_tower = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

            # Policy head
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels, 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * SIZE * SIZE, SIZE * SIZE),
            )

            # Value head
            self.value_head = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(SIZE * SIZE, 64),
                nn.ReLU(),
                # Final layer outputs a single scalar in [-1, 1]
                nn.Linear(64, 1),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.start_conv(x)
            h = self.res_tower(h)
            policy_logits = self.policy_head(h)
            value = self.value_head(h).squeeze(-1)
            return policy_logits, value
else:
    class TinyPolicyValueNet:  # pragma: no cover
        def __init__(self, channels: int = 64, num_blocks: int = 6) -> None:
            raise RuntimeError("PyTorch is required for TinyPolicyValueNet.")


def encode_game_tensor(game: GoGame) -> torch.Tensor:
    """Return [1, 3, 9, 9] float tensor from side-to-move perspective."""
    require_torch()
    board = torch.tensor(game.board, dtype=torch.float32)
    me = game.to_move
    opp = WHITE if me == BLACK else BLACK
    
    p0 = (board == me).float().view(SIZE, SIZE)
    p1 = (board == opp).float().view(SIZE, SIZE)
    p2 = torch.ones((SIZE, SIZE), dtype=torch.float32)
    return torch.stack([p0, p1, p2]).unsqueeze(0)


def encode_games_batch(games: list[GoGame]) -> torch.Tensor:
    """Return [N, 3, 9, 9] float tensor for a batch of games."""
    require_torch()
    if not games:
        return torch.empty((0, 3, SIZE, SIZE), dtype=torch.float32)
    
    # Using a list comprehension for boards is still necessary as they are in GoGame objects,
    # but we move the tensor heavy lifting to vectorized operations.
    boards = torch.tensor([g.board for g in games], dtype=torch.float32)  # [N, 81]
    me_colors = torch.tensor([g.to_move for g in games], dtype=torch.float32).view(-1, 1)
    opp_colors = torch.where(me_colors == BLACK, torch.tensor(WHITE, dtype=torch.float32), torch.tensor(BLACK, dtype=torch.float32))

    p0 = (boards == me_colors).float().view(-1, SIZE, SIZE)
    p1 = (boards == opp_colors).float().view(-1, SIZE, SIZE)
    p2 = torch.ones((len(games), SIZE, SIZE), dtype=torch.float32)
    
    return torch.stack([p0, p1, p2], dim=1)


class PolicyValueModel:
    """Thin wrapper for inference and checkpoint loading."""

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
        probs_list, values = self.predict_batch([game])
        return probs_list[0], values[0]

    def predict_batch(self, games: list[GoGame]) -> tuple[list[list[float]], list[float]]:
        if not games:
            return [], []
        with torch.no_grad():
            x = encode_games_batch(games).to(self.device)
            logits, values = self.net(x)
            # Softmax across the move dimension (dim=1 of [N, 81])
            probs = torch.softmax(logits, dim=1).cpu().tolist()
            vs = values.cpu().tolist()
        return probs, vs
